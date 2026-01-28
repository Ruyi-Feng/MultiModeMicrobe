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
from torch.optim.lr_scheduler import LambdaLR

from exp.train import (
    setup_logger,
    load_train_data,
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
)
from config import train_args
from model import MicrobeProteinRepr, GumbalSoftmax, AttentionConvergence
from model.property_encoder import get_numerical_property_encoder
from utils.tools import get_optimizer_params


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


class E2EPrediction(nn.Module):
    def __init__(self,
                 aa_encoder,
                 hidden_size: int = 128
                 ):
        super(E2EPrediction, self).__init__()
        self.hidden_size = hidden_size
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )  # 目标是预测一个dim=3的向量，分别代表[ph_norm, salt_norm, temp_norm]
        self._init_aa_net(aa_encoder)

    def _init_aa_net(self, aa_encoder):
        self._aa_proj = nn.Linear(aa_encoder.embed_dim, self.hidden_size, bias=False)
        nn.init.xavier_uniform_(self._aa_proj.weight, gain=1.0)
        self.aa_encoder = aa_encoder
        # microbe_protein_repr 需要 cls token，与 backbone 一致
        if getattr(aa_encoder, "name", None) == "microbe_protein_repr":
            self.protein_cls_token = nn.Parameter(
                torch.randn(1, 1, aa_encoder.embed_dim) * 0.02
            )

    def forward(self, aa_seq, padding_mask=None):
        # 与 backbone 一致：microbe_protein_repr 先拼 cls token
        if getattr(self.aa_encoder, "name", None) == "microbe_protein_repr":
            cls_token = self.protein_cls_token.expand(aa_seq.size(0), -1, -1)
            aa_seq = torch.cat([cls_token, aa_seq], dim=1)
            if padding_mask is not None:
                cls_mask = torch.ones(
                    aa_seq.size(0), 1,
                    dtype=padding_mask.dtype,
                    device=padding_mask.device,
                )
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        if getattr(self.aa_encoder, "name", None) == "attention_convergence" and padding_mask is not None:
            aa_embedding, _ = self.aa_encoder(aa_seq, padding_mask=padding_mask, return_weights=True)
        else:
            aa_embedding = self.aa_encoder(aa_seq)

        aa_embedding = self._aa_proj(aa_embedding)
        return self.pred_layer(aa_embedding)

    def get_loss(self, logits, gts):
        # gts is 3d tensor [ph_norm, salt_norm, temp_norm]，若为4维则取前3维
        if gts.size(-1) > 3:
            gts = gts[:, :3]
        return F.mse_loss(logits, gts)


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


def main():
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

        model = model.to(args.device)
        model.train()
        end = time.time()

        for i, batch_data in enumerate(train_loader):
            batch_p, batch_a, padding_mask, batch_keys = batch_data
            padding_mask = padding_mask.to(args.device)
            batch_a = batch_a.to(args.device)

            pred = model(batch_a, padding_mask=padding_mask)

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

    logger.info("E2E Prediction training completed!")


if __name__ == "__main__":
    main()
