# 这个工作独立了e2eprediction结构，直接通过aaencoder聚合之后的表征加预测头预测三个数值。
# 不需要做clip。但是数据加载需要从train里获取。
# 请补全这个脚本。

import os
import shutil
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from exp.train import (
    setup_logger,
    load_train_data,
    collate_fn_individual,
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
)
from config import train_args
from exp.dataset import IndividualDataset, CollectiveDataset
from model import E2EPrediction, MicrobeProteinRepr, GumbalSoftmax, AttentionConvergence
from model.property_encoder import get_numerical_property_encoder
from utils.tools import get_optimizer_params
from utils.json_loader import load_json, save_json


def load_aa_encoder(args):
    """仅构建 aa_encoder，用于 E2E 预测（与 train 中 load_model 的 aa 部分一致）。"""
    if args.aa_encoder_type == "microbe_protein_repr":
        aa_encoder = MicrobeProteinRepr(
            embed_dim=args.aa_repr_dim,
            num_layers=args.aa_encoder_num_layers,
            num_heads=args.aa_encoder_num_heads,
            dropout=args.aa_encoder_dropout,
        )
    elif args.aa_encoder_type == "gumbal_softmax":
        aa_encoder = GumbalSoftmax(embed_dim=args.aa_repr_dim)
    elif args.aa_encoder_type == "attention_convergence":
        aa_encoder = AttentionConvergence(
            embed_dim=args.aa_repr_dim,
            hidden_dim=getattr(args, "aa_encoder_hidden_dim", args.aa_repr_dim),
        )
    else:
        raise ValueError(f"Invalid aa_encoder_type: {args.aa_encoder_type}")
    return aa_encoder


def _visual_weights(aa_weights, keys, save_dir):
    """
    把 aa_weights 画成小方格，颜色是重要性。方格随 seq 长度换行。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(aa_weights, torch.Tensor):
        weights_np = aa_weights.detach().cpu().numpy()
    else:
        weights_np = aa_weights

    for idx, key in enumerate(keys):
        w = weights_np[idx]
        nonzero_idx = np.where(w > 1e-8)[0]
        if len(nonzero_idx) > 0:
            w = w[:nonzero_idx[-1] + 1]

        seq_len = len(w)
        if seq_len == 0:
            continue

        n_cols = 50
        n_rows = math.ceil(seq_len / n_cols)

        pad_len = n_rows * n_cols - seq_len
        w_padded = np.pad(w, (0, pad_len), constant_values=np.nan)
        w_matrix = w_padded.reshape(n_rows, n_cols)

        plt.figure(figsize=(15, max(2, n_rows * 0.5)))
        cmap = plt.cm.viridis
        cmap.set_bad('white')
        im = plt.imshow(w_matrix, cmap=cmap, aspect='equal')
        plt.colorbar(im, label='Importance Score', fraction=0.046, pad=0.04)
        plt.title(f"Protein: {key}")
        plt.axis('off')

        save_path = os.path.join(save_dir, f"{key}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()


def _get_true_index(rank_k: int, protein_index: list):
    if rank_k < len(protein_index):
        if rank_k in protein_index[rank_k]:
            return rank_k
        for i in range(1, rank_k + 1):
            current_idx = rank_k - i
            if rank_k in protein_index[current_idx]:
                return current_idx
        return None
    for i in range(len(protein_index)):
        idx = len(protein_index) - 1 - i
        if rank_k in protein_index[idx]:
            return idx
    return None


def get_importance_score(aa_weights, keys, args, protein_index, top_k=30, if_visualize=False):
    if args.collective:
        raise ValueError("Collective mode is not supported for importance score calculation")

    import_rank = aa_weights.argsort(dim=1, descending=False)
    if if_visualize:
        save_dir = os.path.join(args.save_path, "visual_weights")
        _visual_weights(aa_weights, keys, save_dir)

    batch_top_k_protein_id = {}
    for batch_key, batch_rank_k in zip(keys, import_rank.tolist()):
        top_k_protein_id = {}
        h5_path = os.path.join(args.data_path, f"{batch_key}.h5")
        with h5py.File(h5_path, 'r') as f:
            for i, k in enumerate(batch_rank_k):
                if k >= top_k:
                    continue
                index_k = _get_true_index(i, protein_index[batch_key])
                if index_k is None:
                    continue
                protein_id = f['protein_id'][index_k]
                top_k_protein_id.update({k: protein_id.decode('utf-8')})
        batch_top_k_protein_id.update({batch_key: top_k_protein_id})
    return batch_top_k_protein_id


def init_optimizer(args, model):
    """使用适合回归预测任务的 AdamW + warmup + cosine 学习率。"""
    param_groups = get_optimizer_params(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.0,
    )
    warmup_epochs = getattr(args, "warmup_epochs", 2)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epoch - warmup_epochs, 1)
        return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train():
    args = train_args()

    # 设置日志记录器
    logger = setup_logger(args)

    # 记录训练配置信息
    logger.info("=" * 80)
    logger.info("E2E Prediction Training Configuration:")
    logger.info(f"  Mark: {args.mark}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Start epoch: {args.start_epoch}")
    logger.info(f"  Total epochs: {args.epoch}")
    logger.info(f"  AA encoder type: {args.aa_encoder_type}")
    logger.info(f"  Save path: {args.save_path}")
    logger.info("=" * 80)

    train_loader = load_train_data(args)

    _, tokenizer_p = get_numerical_property_encoder(
        args.property_dim,
        getattr(args, "property_embedding_dim", 128),
    )

    aa_encoder = load_aa_encoder(args)
    hidden_size = getattr(args, "hidden_size", args.cross_hidden_size)
    model = E2EPrediction(aa_encoder, hidden_size=hidden_size)

    optimizer, scheduler = init_optimizer(args, model)

    warmup_epochs = getattr(args, "warmup_epochs", 2)
    logger.info(
        f"Learning rate schedule: warmup={warmup_epochs} epochs, "
        f"initial_lr={args.lr}, total_epochs={args.epoch}"
    )

    for epoch in range(args.start_epoch, args.epoch):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{args.epoch}")
        logger.info(f"{'='*80}")

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        mae_meter = AverageMeter("MAE", ":.4f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, mae_meter],
            prefix="Epoch: [{}]".format(epoch),
        )

        if args.collective:
            protein_index = None
        else:
            protein_index = load_json(args.protein_index_path)
        all_top_k_protein_ids = {}

        model = model.to(args.device)
        model.train()
        end = time.time()

        for i, batch_data in enumerate(train_loader):
            batch_p, batch_a, padding_mask, batch_keys = batch_data
            padding_mask = padding_mask.to(args.device)
            batch_a = batch_a.to(args.device)

            out = model(batch_a, padding_mask=padding_mask)

            pred = out["pred"]
            aa_weights = out.get("aa_weights")
            if (not args.collective) and (aa_weights is not None):
                top_k_protein_ids = get_importance_score(
                    aa_weights,
                    batch_keys,
                    args,
                    protein_index,
                    top_k=30,
                    if_visualize=(i == 0),
                )
                all_top_k_protein_ids.update(top_k_protein_ids)

            tgt = tokenizer_p(batch_p, args.device)
            if tgt.size(-1) > 3:
                tgt = tgt[:, :3]
            loss = model.get_loss(pred, tgt)

            with torch.no_grad():
                mae = F.l1_loss(pred, tgt).item()
            mae_meter.update(mae, batch_a.size(0))
            losses.update(loss.item(), batch_a.size(0))

            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = getattr(args, "max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        scheduler.step()

        save_name = os.path.join(args.save_path, f"e2e_checkpoint_{epoch+1}.pth.tar")
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
        if not args.collective:
            top_k_path = os.path.join(args.save_path, f"e2e_top_k_protein_ids_epoch_{epoch+1}.json")
            save_json(all_top_k_protein_ids, top_k_path)
            logger.info(f"Saved top-k protein ids to {top_k_path}")

    logger.info("E2E Prediction training completed!")


def validate():
    args = train_args()
    logger = setup_logger(args)
    logger.info("=" * 80)
    logger.info("E2E Prediction Validation Configuration:")
    logger.info(f"  Mark: {args.mark}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Start epoch: {args.start_epoch}")
    logger.info(f"  Total epochs: {args.epoch}")
    logger.info(f"  AA encoder type: {args.aa_encoder_type}")
    logger.info(f"  Save path: {args.save_path}")
    logger.info("=" * 80)

    # 这里的 data 会直接从 config 被替换为验证的 data。
    # 验证时输出总体 MSE（norm 后）以及反归一化后的三项 MSE。
    # 给出预测值和真实值的分布情况，三个指标三张图。
    if args.collective:
        val_data = CollectiveDataset(args)
        collate_fn = None
    else:
        val_data = IndividualDataset(args)
        collate_fn = collate_fn_individual
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    _, tokenizer_p = get_numerical_property_encoder(
        args.property_dim,
        getattr(args, "property_embedding_dim", 128),
    )

    aa_encoder = load_aa_encoder(args)
    hidden_size = getattr(args, "hidden_size", args.cross_hidden_size)
    model = E2EPrediction(aa_encoder, hidden_size=hidden_size).to(args.device)

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, weights_only=False, map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', 'N/A')})")
    elif args.resume:
        logger.warning(f"=> no checkpoint found at '{args.resume}'")
    else:
        logger.warning("=> args.resume is None, validating with random initialized model")

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("MSE", ":.4e")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix="Val: ",
    )

    all_preds = []
    all_gts = []
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if args.collective:
                batch_p, batch_a, batch_keys = batch_data
                padding_mask = None
            else:
                batch_p, batch_a, padding_mask, batch_keys = batch_data
                padding_mask = padding_mask.to(args.device)
            batch_a = batch_a.to(args.device)

            pred = model(batch_a, padding_mask=padding_mask)
            tgt = tokenizer_p(batch_p, args.device)
            if tgt.size(-1) > 3:
                tgt = tgt[:, :3]

            loss = model.get_loss(pred, tgt)
            losses.update(loss.item(), batch_a.size(0))

            all_preds.append(pred.detach().cpu())
            all_gts.append(tgt.detach().cpu())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i, logger)

    if len(all_preds) == 0:
        logger.warning("No validation data found.")
        return

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    overall_mse = F.mse_loss(preds, gts).item()
    logger.info(f"Overall MSE (normalized): {overall_mse:.6f}")

    def denorm(x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    ph_pred = denorm(preds[:, 0], 1.0, 14.0)
    ph_gt = denorm(gts[:, 0], 1.0, 14.0)
    salt_pred = denorm(preds[:, 1], 0.0, 30.0)
    salt_gt = denorm(gts[:, 1], 0.0, 30.0)
    temp_pred = denorm(preds[:, 2], 0.0, 100.0)
    temp_gt = denorm(gts[:, 2], 0.0, 100.0)

    mse_ph = F.mse_loss(ph_pred, ph_gt).item()
    mse_salt = F.mse_loss(salt_pred, salt_gt).item()
    mse_temp = F.mse_loss(temp_pred, temp_gt).item()
    logger.info(f"Denorm MSE - pH: {mse_ph:.6f}, salt: {mse_salt:.6f}, temp: {mse_temp:.6f}")

    plot_dir = os.path.join(args.save_path, "e2e_val_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        def plot_distribution(pred_vals, gt_vals, name, unit):
            plt.figure(figsize=(6, 4))
            plt.hist(gt_vals, bins=30, alpha=0.6, label="gt")
            plt.hist(pred_vals, bins=30, alpha=0.6, label="pred")
            plt.title(f"{name} distribution")
            plt.xlabel(unit)
            plt.ylabel("count")
            plt.legend()
            save_path = os.path.join(plot_dir, f"{name.lower()}_dist.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved distribution plot: {save_path}")

        plot_distribution(ph_pred.numpy(), ph_gt.numpy(), "pH", "pH")
        plot_distribution(salt_pred.numpy(), salt_gt.numpy(), "Salt", "%")
        plot_distribution(temp_pred.numpy(), temp_gt.numpy(), "Temp", "C")
    except Exception as exc:
        logger.warning(f"Plotting failed: {exc}")


if __name__ == "__main__":
    # train()
    validate()
