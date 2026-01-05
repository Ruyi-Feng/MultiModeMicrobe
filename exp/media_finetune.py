import os
import time
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime

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

    return optimizer

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
    
    return batch_aa, target_ids, attention_mask

def train_epoch(args, clip_model, decoder_model, loader, optimizer, epoch, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    decoder_model.train()
    clip_model.eval() # 始终 Eval
    
    end = time.time()
    
    for i, (batch_aa, target_ids, attention_mask) in enumerate(loader):
        batch_aa = batch_aa.to(args.device)
        target_ids = target_ids.to(args.device)
        # attention_mask = attention_mask.to(args.device)
        
        # 1. Extract Microbe Features
        with torch.no_grad():
            # MicrobeCLIP forward
            # 构造 padding_mask for aa
            # aa_repr: [B, S, D]
            # padding_mask: [B, S] (True for valid)
            padding_mask = (batch_aa.abs().sum(dim=-1) > 0)
            
            # 使用 _forward_aa 接口
            aa_embedding, _ = clip_model._forward_aa(batch_aa, padding_mask=padding_mask)
            
            input_vectors = aa_embedding # [B, Dim]
            
        # 2. Decoder Forward
        outputs = decoder_model(
            input_vectors=input_vectors,
            target_token_ids=target_ids
        )
        
        loss = outputs.loss
        
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip grad
        torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        losses.update(loss.item(), batch_aa.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i, logger)

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
    decoder_model = load_decoder_checkpoint(args, decoder_model, logger)
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
    optimizer = init_finetune_optimizer(args, decoder_model)

    # 5. Training Loop
    logger.info("Start Training...")
    for epoch in range(args.start_epoch, args.epoch):
        train_epoch(args, clip_model, decoder_model, train_loader, optimizer, epoch, logger)

        # Save Checkpoint
        save_path = os.path.join(args.save_path, f"finetune_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': decoder_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
