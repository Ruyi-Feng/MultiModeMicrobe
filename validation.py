import os
import time
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from torch.utils.data import DataLoader

from config import train_args
from exp.dataset import IndividualDataset, CollectiveDataset
from utils.tools import compute_topk_accuracy
from train import (
    collate_fn_individual,
    property_converter,
    load_model,
    clip_accuracy,
    AverageMeter,
    ProgressMeter
)

def setup_logger(args):
    """
    设置日志记录器，同时输出到控制台和文件
    """
    # 确保日志目录存在
    os.makedirs(args.save_path, exist_ok=True)

    # 创建日志文件名，包含 mark 和时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"val_{args.mark}_{timestamp}.log"
    log_filepath = os.path.join(args.save_path, log_filename)

    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 创建 logger
    logger = logging.getLogger('validation')
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

def load_val_data(args):
    if args.collective:
        val_data = CollectiveDataset(args)
        collate_fn = None  # 使用默认collate_fn
    else:
        val_data = IndividualDataset(args)
        collate_fn = collate_fn_individual  # 使用自定义collate_fn处理变长序列

    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,  # 验证集不shuffle
                            collate_fn=collate_fn)
    return val_loader

def validate(args, model, tokenizer_p, loader, logger):
    """
    Batch-wise validation
    """
    logger.info("starting batch-wise validation")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix="Test: "
    )

    model.eval()
    model.to(args.device)

    with torch.no_grad():
        end = time.time()
        for i, batch_data in enumerate(loader):
            # 使用dataloader获取aa和property pairs
            if args.collective:
                batch_p, batch_a = batch_data
                padding_mask = None
            else:
                batch_p, batch_a, padding_mask = batch_data
                padding_mask = padding_mask.to(args.device)

            batch_a = batch_a.to(args.device)
            batch_p, cls_index_p = property_converter(batch_p, tokenizer_p, args.device)

            pred = model(
                batch_a,
                batch_p,
                property_cls_token_index=cls_index_p,
                return_hidden_states=False,
                padding_mask=padding_mask
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        logger.info(f' * Batch-wise Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return top1.avg

def validate_retrieval(args, model, tokenizer_p, loader, logger):
    """
    Full dataset retrieval validation (Global Cosine Similarity)
    """
    logger.info(f"Starting retrieval validation on full dataset (Size: {len(loader.dataset)})...")
    model.eval()
    model.to(args.device)

    all_aa_feats = []
    all_prop_feats = []

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if args.collective:
                batch_p, batch_a = batch_data
                padding_mask = None
            else:
                batch_p, batch_a, padding_mask = batch_data
                padding_mask = padding_mask.to(args.device)

            batch_a = batch_a.to(args.device)
            batch_p, cls_index_p = property_converter(batch_p, tokenizer_p, args.device)

            # Get embeddings
            out = model(
                batch_a,
                batch_p,
                property_cls_token_index=cls_index_p,
                return_hidden_states=True,
                padding_mask=padding_mask
            )

            # 收集 normalized features (on CPU to save GPU memory)
            all_aa_feats.append(out["aa_representation"].cpu())
            all_prop_feats.append(out["property_representation"].cpu())

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1} batches...")

    # Concatenate all features
    all_aa_feats = torch.cat(all_aa_feats, dim=0)
    all_prop_feats = torch.cat(all_prop_feats, dim=0)

    logger.info(f"Collected features: AA {all_aa_feats.shape}, Property {all_prop_feats.shape}")

    # Compute similarity matrix
    device = args.device

    try:
        # Try full matrix on GPU
        logger.info("Calculating similarity matrix...")
        all_aa_feats = all_aa_feats.to(device)
        all_prop_feats = all_prop_feats.to(device)
        logit_scale = model.logit_scale.exp().to(device)

        # Logits = scale * AA @ Prop.T
        # Note: Features should already be normalized by the model
        logits = logit_scale * all_aa_feats @ all_prop_feats.t()

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("OOM on GPU, switching to CPU for similarity calculation...")
            torch.cuda.empty_cache()
            all_aa_feats = all_aa_feats.cpu()
            all_prop_feats = all_prop_feats.cpu()
            logit_scale = model.logit_scale.exp().cpu()
            logits = logit_scale * all_aa_feats @ all_prop_feats.t()
            device = 'cpu'
        else:
            raise e

    # Generate labels (diagonal)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=device)

    # Calculate Accuracies
    logger.info("Computing top-k accuracy...")
    acc1_a2p, acc5_a2p = compute_topk_accuracy(logits, labels, topk=(1, 5))
    acc1_p2a, acc5_p2a = compute_topk_accuracy(logits.t(), labels, topk=(1, 5))

    logger.info(f"Retrieval Results (Total {batch_size} samples):")
    logger.info(f"AA -> Property: Acc@1: {acc1_a2p.item():.2f}%, Acc@5: {acc5_a2p.item():.2f}%")
    logger.info(f"Property -> AA: Acc@1: {acc1_p2a.item():.2f}%, Acc@5: {acc5_p2a.item():.2f}%")

    avg_acc1 = (acc1_a2p + acc1_p2a) / 2
    avg_acc5 = (acc5_a2p + acc5_p2a) / 2
    logger.info(f"Average: Acc@1: {avg_acc1.item():.2f}%, Acc@5: {avg_acc5.item():.2f}%")

    return avg_acc1.item()

def main():
    args = train_args()

    # 设置日志记录器
    logger = setup_logger(args)

    logger.info("=" * 80)
    logger.info("Validation Configuration:")
    logger.info(f"  Mark: {args.mark}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Index path: {args.index_path}")
    logger.info(f"  Protein index path: {args.protein_index_path}")
    logger.info(f"  Resume checkpoint: {args.resume}")
    logger.info("=" * 80)

    val_loader = load_val_data(args)
    model, property_tokenizer = load_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logger.warning(f"=> no checkpoint found at '{args.resume}'")
    else:
        logger.warning("No checkpoint provided. Validation will use random weights!")

    # 运行 batch-wise 验证
    # validate(args, model, property_tokenizer, val_loader, logger)

    # 运行 retrieval 验证
    validate_retrieval(args, model, property_tokenizer, val_loader, logger)

if __name__ == "__main__":
    main()
