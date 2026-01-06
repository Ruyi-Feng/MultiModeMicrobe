import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

from utils.media_match import parse_medium_recipe, calculate_recipe_loss
from config import train_args
from exp.dataset import MediaDataset
from model.media_decoder import MediaDecoder
from exp.train import load_model, AverageMeter, ProgressMeter

def setup_logger(args):
    os.makedirs(args.save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"finetune_{args.mark}_{timestamp}.log"
    log_filepath = os.path.join(args.save_path, log_filename)

    logger = logging.getLogger('finetune')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger

def init_finetune_optimizer(args, model):
    """
    初始化优化器，针对 LoRA 参数使用不同的学习率。
    参考 train.py 的实现。
    """
    # 检查是否使用 LoRA (根据参数名判断)
    # 在 MediaDecoder 中，LoRA 参数包含 'lora_'
    # Projector 参数不包含 'lora_'
    
    lora_lr_multiplier = getattr(args, 'lora_lr_multiplier', 10.0)
    
    lora_decay_params = []
    lora_no_decay_params = []
    other_decay_params = []
    other_no_decay_params = []
    no_decay = ["bias", "LayerNorm.weight", "layernorm"]

    trainable_params_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        trainable_params_count += 1

        is_lora = 'lora_' in name.lower()
        has_no_decay = any(nd in name for nd in no_decay)

        if is_lora:
            if has_no_decay:
                lora_no_decay_params.append(param)
            else:
                lora_decay_params.append(param)
        else:
            if has_no_decay:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)

    print(f"Total trainable parameters found: {trainable_params_count}")
    
    param_groups = []
    
    # LoRA Groups (Higher LR)
    if lora_decay_params:
        param_groups.append({
            "params": lora_decay_params,
            "lr": args.lr * lora_lr_multiplier,
            "weight_decay": args.weight_decay
        })
    if lora_no_decay_params:
        param_groups.append({
            "params": lora_no_decay_params,
            "lr": args.lr * lora_lr_multiplier,
            "weight_decay": 0.0
        })
        
    # Other Groups (Base LR, e.g. Projector)
    if other_decay_params:
        param_groups.append({
            "params": other_decay_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        })
    if other_no_decay_params:
        param_groups.append({
            "params": other_no_decay_params,
            "lr": args.lr,
            "weight_decay": 0.0
        })

    print(f"Optimizer groups:")
    print(f"  LoRA params (LR x{lora_lr_multiplier}): {len(lora_decay_params) + len(lora_no_decay_params)}")
    print(f"  Other params (Base LR): {len(other_decay_params) + len(other_no_decay_params)}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.0 # Default, overridden by groups
    )

    # 添加学习率调度器（warmup + cosine annealing）
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

def collate_fn_media(batch, tokenizer):
    """
    batch: [(description, aa_representation, key, media_md), ...]
    """
    descriptions, aa_reprs, keys, media_mds = zip(*batch)

    # 1. 处理 AA Representation (复用 individual 的逻辑)
    # 找到最大序列长度
    max_seq_len = 0
    valid_reprs = [x for x in aa_reprs if x is not None]
    if valid_reprs:
        max_seq_len = max(x.shape[0] for x in valid_reprs)
    else:
        max_seq_len = 1 # Fallback

    padded_aa_reprs = []
    seq_lengths = []

    for aa_repr in aa_reprs:
        if aa_repr is None:
            if valid_reprs:
                dim = valid_reprs[0].shape[1]
                aa_repr = torch.zeros(max_seq_len, dim)
                seq_len = 0
            else:
                 # 极端情况
                aa_repr = torch.zeros(max_seq_len, 64) # 假设 dim
                seq_len = 0
        else:
            seq_len, dim = aa_repr.shape
            if seq_len > max_seq_len:
                aa_repr = aa_repr[:max_seq_len]
                seq_len = max_seq_len
            elif seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, dim, dtype=aa_repr.dtype)
                aa_repr = torch.cat([aa_repr, padding], dim=0)
        
        padded_aa_reprs.append(aa_repr)
        seq_lengths.append(seq_len)
        
    batch_aa = torch.stack(padded_aa_reprs, dim=0)
    
    # 2. 处理 Media MD Text -> Token IDs
    # 添加 EOS token
    texts = [t + tokenizer.eos_token for t in media_mds]
    
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024, # 限制最大长度防止 OOM
        add_special_tokens=False # 我们手动控制 special tokens
    )
    
    target_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask # 用于 Decoder 的 mask (虽然 Qwen 可能主要看 labels)
    
    return batch_aa, target_ids, attention_mask, list(media_mds)

def get_media_loss(outputs, target_ids, media_mds, tokenizer, recipe_loss_weight=0.1):
    """
    计算media loss，包括output loss和recipe loss
    :param outputs: 模型输出，包含loss和logits
    :param target_ids: 目标token ids [batch_size, seq_len]
    :param media_mds: 真实的markdown文本列表
    :param tokenizer: tokenizer用于解码
    :param recipe_loss_weight: recipe loss的权重，用于平衡两种loss
    :return: 总loss (output_loss + recipe_loss_weight * recipe_loss)
    """
    
    # 1. 获取output loss（模型的标准语言模型loss）
    output_loss = outputs.loss

    # 获取预测的token ids（argmax）
    pred_token_ids = torch.argmax(outputs.logits, dim=-1)  # [batch_size, seq_len]

    batch_size = pred_token_ids.shape[0]
    target_len = target_ids.shape[1]

    total_seq_len = outputs.logits.shape[1]
    target_start_idx = max(0, total_seq_len - target_len)

    if target_start_idx >= total_seq_len:
        pred_target_ids = pred_token_ids
    else:
        pred_target_ids = pred_token_ids[:, target_start_idx:]  # [batch_size, target_len]

    recipe_losses = []

    for i in range(batch_size):
        # 解码预测的token ids
        pred_ids = pred_target_ids[i]
        # 移除padding tokens和EOS tokens
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        
        # 找到第一个EOS token的位置，只保留EOS之前的部分
        eos_positions = (pred_ids == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # 取第一个EOS之前的部分
            valid_length = eos_positions[0].item()
            valid_pred_ids = pred_ids[:valid_length]
        else:
            # 如果没有EOS，移除padding tokens
            valid_pred_ids = pred_ids[pred_ids != pad_token_id]
        
        if len(valid_pred_ids) > 0:
            pred_text = tokenizer.decode(valid_pred_ids, skip_special_tokens=False)
        else:
            pred_text = ""
        
        # 提取assistant回复部分（参考media_inference的方式）
        if "<|im_start|>assistant\n" in pred_text:
            predicted_text = pred_text.split("<|im_start|>assistant\n")[-1]
            # 移除可能的结束标记
            predicted_text = predicted_text.replace("<|im_end|>", "").strip()
        else:
            predicted_text = pred_text.strip()
        
        # 获取真实的md text
        gt_md_text = media_mds[i]
        
        # 5. 解析成dict并计算recipe loss
        try:
            pred_dict = parse_medium_recipe(predicted_text)
            gt_dict = parse_medium_recipe(gt_md_text)
            recipe_loss = calculate_recipe_loss(gt_dict, pred_dict)
            # 限制recipe_loss的最大值，避免过大惩罚
            recipe_loss = min(recipe_loss, 50.0)  # 设置上限
            recipe_losses.append(recipe_loss)
        except Exception as e:
            # 如果解析失败，使用一个合理的惩罚值（降低惩罚）
            recipe_losses.append(50.0)
    
    # 6. 计算平均recipe loss并转换为tensor
    avg_recipe_loss = sum(recipe_losses) / len(recipe_losses) if recipe_losses else 0.0
    recipe_loss_tensor = torch.tensor(avg_recipe_loss, device=output_loss.device, dtype=output_loss.dtype, requires_grad=False)
    
    # 7. 最终loss = output loss + recipe_loss_weight * recipe loss
    # 注意：recipe_loss不可微分，只用于监控，不影响梯度
    # 但通过权重控制，避免recipe_loss过大掩盖output_loss的变化
    total_loss = output_loss + recipe_loss_weight * recipe_loss_tensor
    
    return total_loss, output_loss.item(), avg_recipe_loss

def train_epoch(args, clip_model, decoder_model, loader, optimizer, scheduler, epoch, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    output_losses = AverageMeter("OutputLoss", ":.4e")
    recipe_losses = AverageMeter("RecipeLoss", ":.4e")
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, output_losses, recipe_losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    decoder_model.train()
    if args.freeze_aa_encoder:
        clip_model.eval()
    else:
        clip_model.train()

    end = time.time()

    for i, (batch_aa, target_ids, _, media_mds) in enumerate(loader):
        batch_aa = batch_aa.to(args.device)
        target_ids = target_ids.to(args.device)

        # 1. Extract Microbe Features
        with torch.no_grad():
            padding_mask = (batch_aa.abs().sum(dim=-1) > 0)
            aa_embedding, _ = clip_model._forward_aa(batch_aa, padding_mask=padding_mask)

            input_vectors = aa_embedding # [B, Dim]

        # 2. Decoder Forward
        outputs = decoder_model(
            input_vectors=input_vectors,
            target_token_ids=target_ids
        )

        # 3. Calculate Loss (output loss + recipe loss)
        recipe_loss_weight = getattr(args, 'recipe_loss_weight', 0.1)
        loss, output_loss_val, recipe_loss_val = get_media_loss(
            outputs, target_ids, media_mds, decoder_model.tokenizer, recipe_loss_weight
        )

        # 4. Backward
        optimizer.zero_grad()
        loss.backward()

        # Clip grad
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), args.max_grad_norm)

        optimizer.step()

        losses.update(loss.item(), batch_aa.size(0))
        output_losses.update(output_loss_val, batch_aa.size(0))
        recipe_losses.update(recipe_loss_val, batch_aa.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i, logger)
            # 添加调试信息
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Grad norm: {grad_norm:.4f}, LR: {current_lr:.2e}, "
                       f"Output Loss: {output_loss_val:.4f}, Recipe Loss: {recipe_loss_val:.4f}")

def load_encoder(args, logger):
    if not args.resume:
        logger.warning("No --resume checkpoint provided for MicrobeCLIP! Random weights will be used (Not Recommended).")

    clip_model, _ = load_model(args)
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading MicrobeCLIP from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        clip_model.load_state_dict(checkpoint['state_dict'], strict=False) # strict=False 以防版本差异

    clip_model.to(args.device)
    return clip_model

def load_decoder(args, logger):
    decoder_model = MediaDecoder(
        qwen_model_path=args.decoder_model_name, # 可以改为参数传入
        input_vector_dim=args.cross_hidden_size,
        lora_rank=args.lora_rank_decoder,
        lora_alpha=args.lora_alpha_decoder,
        lora_dropout=args.lora_dropout_decoder
    )
    load_decoder_checkpoint(args, decoder_model, logger)
    decoder_model.to(args.device)
    logger.info(f"Decoder model initialized with input dim {args.cross_hidden_size}")
    return decoder_model

def load_decoder_checkpoint(args, decoder_model, logger):
    """
    加载 decoder 的 checkpoint
    """
    if not hasattr(args, 'decoder_checkpoint') or not args.decoder_checkpoint:
        logger.warning("未提供 decoder checkpoint，使用随机初始化的权重")
        return

    checkpoint_path = args.decoder_checkpoint
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"Checkpoint 文件不存在: {checkpoint_path}，使用原始模型权重")
        return

    logger.info(f"加载 Decoder checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        decoder_model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"从 checkpoint 加载 state_dict (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        decoder_model.load_state_dict(checkpoint, strict=False)
        logger.info("从 checkpoint 加载 state_dict")


def load_networks(args, logger):

    encoder_model = load_encoder(args, logger)

    decoder_model = load_decoder(args, logger)

    return encoder_model, decoder_model


def main():
    args = train_args()
    logger = setup_logger(args)
    logger.info("Initializing Media Finetuning...")
    logger.info(f"Device: {args.device}")

    clip_model, decoder_model = load_networks(args, logger)

    logger.info("Loading Media Dataset...")
    dataset = MediaDataset(args)

    collate_fn = partial(collate_fn_media, tokenizer=decoder_model.tokenizer)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True
    )

    # 使用专门的优化器初始化函数
    optimizer, scheduler = init_finetune_optimizer(args, decoder_model)
    
    # 记录训练配置
    recipe_loss_weight = getattr(args, 'recipe_loss_weight', 0.1)
    logger.info(f"Recipe loss weight: {recipe_loss_weight}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LoRA LR multiplier: {getattr(args, 'lora_lr_multiplier', 10.0)}")
    logger.info(f"Warmup epochs: {getattr(args, 'warmup_epochs', 2)}")

    # 5. Training Loop
    logger.info("Start Training...")
    for epoch in range(args.start_epoch, args.epoch):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.epoch}, Current LR: {current_lr:.2e}")
        
        train_epoch(args, clip_model, decoder_model, train_loader, optimizer, scheduler, epoch, logger)
        
        # 更新学习率
        scheduler.step()

        # Save Checkpoint
        dec_save_path = os.path.join(args.save_path, f"finetune_decoder_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': decoder_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, dec_save_path)
        if not args.freeze_aa_encoder:
            enc_save_path = os.path.join(args.save_path, f"finetune_encoder_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': clip_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, enc_save_path)
        logger.info(f"Saved checkpoint to {dec_save_path}")
