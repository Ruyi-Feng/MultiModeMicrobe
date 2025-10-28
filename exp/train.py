import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from exp.dataset import IndividualDataset, CollectiveDataset
from model import MicrobeCLIP, MicrobeProteinRepr
from model.property_encoder import get_property_encoder
from utils.tools import get_optimizer_params, compute_topk_accuracy
from config import train_args


def load_train_data(args):
    if args.collective:
        train_data = CollectiveDataset(args)
    else:
        train_data = IndividualDataset(args)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True)
    return train_loader

def property_converter(property_seq, property_tokenizer, device="cuda"):
    seq_with_cls = ["<|im_start|> " + seq + "<|endoftext|>" for seq in property_seq]
    property_seq = property_tokenizer(
        seq_with_cls,
        return_tensors='pt',            # 返回 PyTorch tensor
        padding=True,                   # 自动 padding 到最长序列
        truncation=True,                # 超长截断
        max_length=512                  # 可选
    )

    property_seq = {k: v.to(device) for k, v in property_seq.items()}
    tail_index =  (property_seq['attention_mask'].sum(1) - 1)

    return property_seq, tail_index

def load_model(args):
    trainable = {
        'aa_encoder': (not args.freeze_aa_encoder),
        'property_encoder': (not args.freeze_property_encoder),
        'llm_decoder': (not args.freeze_llm_decoder),
                 }
    property_encoder, property_tokenizer = get_property_encoder(args.property_model_path, args.device)
    if args.collective:
        aa_encoder = MicrobeProteinRepr(embed_dim=args.aa_encoder_hidden_size,
                                        num_heads=args.aa_encoder_num_heads,
                                        dropout=args.aa_encoder_dropout,)
        backbone = MicrobeCLIP(property_encoder,
                               trainable=trainable,
                               aa_encoder=aa_encoder,
                               collective=args.collective)
    else:
        backbone = MicrobeCLIP(property_encoder,
                               trainable=trainable,
                               collective=args.collective)
    return backbone, property_tokenizer

def init_optimizer(args, model):
    """
    CLIP AdamW 优化器
    We use the Adam optimizer (Kingma & Ba, 2014)
    with decoupled weight decay regularization (Loshchilov & Hutter, 2017)
    applied to all weights that are not gains or biases,
    and decay the learning rate using a cosine schedule
    (Loshchilov & Hutter, 2016).
    """
    param_groups = get_optimizer_params(model, args.weight_decay)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.lr * 0.01
        )
    return optimizer, scheduler

def train(args, model, tokenizer_p, loader, optimizer, epoch):
    """
    这里接收的都是定义好的model, device, loader, optimizer
    其中loader是对比学习的loader,已经直接加载了对比学习的mini batch的
    每个loader传递的数据都已经加过了cls token, cls token的位置包含在data_config里
    经过了tokenizer
    """
    print("starting training")
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.to(args.device)
    model.train()

    for i, batch_data in enumerate(loader):
        # 使用dataloader获取aa和property pairs
        batch_a, batch_p = batch_data
        # aa B, S
        # property item B, S, H

        batch_a = batch_a.to(args.device)
        batch_p, cls_index_p = property_converter(batch_p, tokenizer_p, args.device)

        pred = model(
            batch_a,
            batch_p,
            property_cls_token_index=cls_index_p,
            return_hidden_states=False
            )

        logits_a = pred["logits_aa"]
        logits_p = pred["logits_property"]

        N = logits_a.shape[0]
        labels = torch.arange(N).to(args.device)
        loss_a = F.cross_entropy(logits_a, labels)
        loss_p = F.cross_entropy(logits_p, labels)
        loss = (loss_a + loss_p) / 2

        acc1, acc5 = clip_accuracy(loss_a, loss_p, topk=(1, 5))
        losses.update(loss.item(), batch_p.size(0))
        top1.update(acc1[0], batch_p.size(0))
        top5.update(acc5[0], batch_p.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def main():
    args = train_args()
    train_loader = load_train_data(args)
    model, property_tokenizer = load_model(args)
    optimizer, scheduler = init_optimizer(args, model)

    for epoch in range(args.start_epoch, args.epochs):

        scheduler.step()
        train(args, model, property_tokenizer, train_loader, optimizer, epoch)

        save_name = os.path.join(args.save_path, "checkpoint.pth.tar")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename=save_name,
        )

def save_checkpoint(state, is_best, filename: str = "checkpoint.pth.tar") -> None:
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

@torch.no_grad()
def clip_accuracy(logits_aa, logits_property, topk=(1, 5)):
    """
    Compute top-k accuracy for CLIP-style contrastive training.
    Each row i of logits_per_image corresponds to similarities between
    image i and all text samples, and vice versa.
    """
    batch_size = logits_aa.size(0)
    labels = torch.arange(batch_size, device=logits_aa.device)

    acc_a2p = compute_topk_accuracy(logits_aa, labels, topk)
    acc_p2a = compute_topk_accuracy(logits_property, labels, topk)

    # 平均两个方向的结果
    acc = [(a + b) / 2 for a, b in zip(acc_a2p, acc_p2a)]
    return acc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
