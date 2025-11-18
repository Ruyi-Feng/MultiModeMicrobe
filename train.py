import os
import shutil
import time
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from exp.dataset import IndividualDataset, CollectiveDataset
from model import MicrobeCLIP, MicrobeProteinRepr, GumbalSoftmax, AttentionConvergence
from model.property_encoder import get_property_encoder
from utils.tools import get_optimizer_params, compute_topk_accuracy
from config import train_args


def setup_logger(args):
    """
    设置日志记录器，同时输出到控制台和文件
    """
    # 确保日志目录存在
    os.makedirs(args.save_path, exist_ok=True)

    # 创建日志文件名，包含 mark 和时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{args.mark}_{timestamp}.log"
    log_filepath = os.path.join(args.save_path, log_filename)

    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 创建 logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # 清除已有的处理器，避免重复添加
    logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger

def collate_fn_individual(batch):
    """
    自定义collate函数，处理individual模式下的变长序列
    """
    descriptions, aa_reprs = zip(*batch)

    # 找到最大序列长度
    max_seq_len = max(aa_repr.shape[0] for aa_repr in aa_reprs if aa_repr is not None)

    # 对每个aa_repr进行padding或truncation
    padded_aa_reprs = []
    for aa_repr in aa_reprs:
        if aa_repr is None:
            # 如果为None，创建一个零张量
            if len(padded_aa_reprs) > 0:
                dim = padded_aa_reprs[0].shape[1]
                aa_repr = torch.zeros(max_seq_len, dim, dtype=torch.float32)
            else:
                raise ValueError("Cannot determine feature dimension from None tensor")
        else:
            seq_len, dim = aa_repr.shape
            if seq_len > max_seq_len:
                # 截断
                aa_repr = aa_repr[:max_seq_len]
            elif seq_len < max_seq_len:
                # Padding
                padding = torch.zeros(max_seq_len - seq_len, dim, dtype=aa_repr.dtype)
                aa_repr = torch.cat([aa_repr, padding], dim=0)
        padded_aa_reprs.append(aa_repr)

    # 堆叠成batch
    batch_aa = torch.stack(padded_aa_reprs, dim=0)
    batch_descriptions = list(descriptions)

    return batch_descriptions, batch_aa

def load_train_data(args):
    if args.collective:
        train_data = CollectiveDataset(args)
        collate_fn = None  # 使用默认collate_fn
    else:
        train_data = IndividualDataset(args)
        collate_fn = collate_fn_individual  # 使用自定义collate_fn处理变长序列

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
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

def resume(args, model, optimizer, logger):
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(args.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.warning("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer

def load_model(args):
    trainable = {
        'aa_encoder': (not args.freeze_aa_encoder),
        'property_encoder': (not args.freeze_property_encoder),
        'llm_decoder': (not args.freeze_llm_decoder),
                 }
    property_encoder, property_tokenizer = get_property_encoder(args.property_model_path, args.device)
    if args.collective:
        backbone = MicrobeCLIP(property_encoder,
                               trainable=trainable,
                               collective=args.collective,
                               cross_hidden_size=args.cross_hidden_size,
                               aa_representation_dim=args.aa_repr_dim)
    else:
        if args.aa_encoder_type == "microbe_protein_repr":
            aa_encoder = MicrobeProteinRepr(embed_dim=args.aa_repr_dim,
                                        num_layers=args.aa_encoder_num_layers,
                                        num_heads=args.aa_encoder_num_heads,
                                        dropout=args.aa_encoder_dropout,)
        elif args.aa_encoder_type == "gumbal_softmax":
            aa_encoder = GumbalSoftmax(embed_dim=args.aa_repr_dim, hidden_dim=args.aa_encoder_hidden_dim)
        elif args.aa_encoder_type == "attention_convergence":
            aa_encoder = AttentionConvergence(embed_dim=args.aa_repr_dim)
        else:
            raise ValueError(f"Invalid aa_encoder_type: {args.aa_encoder_type}")
        backbone = MicrobeCLIP(property_encoder,
                               trainable=trainable,
                               aa_encoder=aa_encoder,
                               collective=args.collective,
                               cross_hidden_size=args.cross_hidden_size,
                               aa_representation_dim=args.aa_repr_dim)
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
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )

    # 添加 warmup 的 scheduler
    warmup_epochs = getattr(args, 'warmup_epochs', 2)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: 线性增长
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing: 从1.0衰减到0.01
            progress = (epoch - warmup_epochs) / max(args.epoch - warmup_epochs, 1)
            return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def train(args, model, tokenizer_p, loader, optimizer, epoch, logger):
    """
    这里接收的都是定义好的model, device, loader, optimizer
    其中loader是对比学习的loader,已经直接加载了对比学习的mini batch的
    每个loader传递的数据都已经加过了cls token, cls token的位置包含在data_config里
    经过了tokenizer
    """
    logger.info("starting training")
    batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.to(args.device)
    model.train()
    end = time.time()

    for i, batch_data in enumerate(loader):
        # 使用dataloader获取aa和property pairs
        batch_p, batch_a = batch_data
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

        acc1, acc5 = clip_accuracy(logits_a, logits_p, topk=(1, 5))
        losses.update(loss.item(), batch_a.size(0))
        top1.update(acc1.item(), batch_a.size(0))
        top5.update(acc5.item(), batch_a.size(0))

        optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪，防止梯度爆炸
        max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # 添加调试信息：每print_freq个batch打印一次详细统计
        if i % args.print_freq == 0:
            progress.display(i, logger)
            # 打印额外的调试信息
            # with torch.no_grad():
            #     logit_scale_val = model.logit_scale.exp().item()
            #     logits_a_mean = logits_a.mean().item()
            #     logits_a_std = logits_a.std().item()
            #     logits_p_mean = logits_p.mean().item()
            #     logits_p_std = logits_p.std().item()
            #     current_lr = optimizer.param_groups[0]['lr']
                # logger.info(f"  Debug: logit_scale={logit_scale_val:.4f}, "
                #       f"logits_aa=[mean={logits_a_mean:.4f}, std={logits_a_std:.4f}], "
                #       f"logits_prop=[mean={logits_p_mean:.4f}, std={logits_p_std:.4f}], "
                #       f"grad_norm={grad_norm:.4f}, lr={current_lr:.2e}")


def main():
    args = train_args()

    # 设置日志记录器
    logger = setup_logger(args)

    # 记录训练配置信息
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Mark: {args.mark}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Start epoch: {args.start_epoch}")
    logger.info(f"  Total epochs: {args.epoch}")
    logger.info(f"  Collective mode: {args.collective}")
    logger.info(f"  AA encoder type: {args.aa_encoder_type}")
    logger.info(f"  Save path: {args.save_path}")
    logger.info("=" * 80)

    train_loader = load_train_data(args)
    model, property_tokenizer = load_model(args)
    optimizer, scheduler = init_optimizer(args, model)

    # 记录学习率调度信息
    warmup_epochs = getattr(args, 'warmup_epochs', 2)
    logger.info(f"Learning rate schedule: warmup={warmup_epochs} epochs, "
          f"initial_lr={args.lr}, total_epochs={args.epoch}")

    model, optimizer = resume(args, model, optimizer, logger)

    for epoch in range(args.start_epoch, args.epoch):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{args.epoch}")
        logger.info(f"{'='*80}")

        scheduler.step()
        train(args, model, property_tokenizer, train_loader, optimizer, epoch, logger)

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
        logger.info(f"Checkpoint saved to {save_name}")

    logger.info("Training completed!")

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

    def display(self, batch, logger) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
