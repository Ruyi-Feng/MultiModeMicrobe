import os
import torch
from torch.utils.data import DataLoader
from functools import partial

from config import train_args
from exp.dataset import MediaDataset
from exp.media_finetune import setup_logger, load_networks

def collate_fn_media_inference(batch, tokenizer):
    """
    推理用的 collate 函数，不需要 target_ids
    batch: [(description, aa_representation, key, media_md), ...]
    返回: batch_aa, keys, media_mds
    """
    descriptions, aa_reprs, keys, media_mds = zip(*batch)
    
    # 处理 AA Representation (复用训练时的逻辑)
    max_seq_len = 0
    valid_reprs = [x for x in aa_reprs if x is not None]
    if valid_reprs:
        max_seq_len = max(x.shape[0] for x in valid_reprs)
    else:
        max_seq_len = 1
    
    padded_aa_reprs = []
    
    for aa_repr in aa_reprs:
        if aa_repr is None:
            if valid_reprs:
                dim = valid_reprs[0].shape[1]
                aa_repr = torch.zeros(max_seq_len, dim)
            else:
                aa_repr = torch.zeros(max_seq_len, 64)
        else:
            seq_len, dim = aa_repr.shape
            if seq_len > max_seq_len:
                aa_repr = aa_repr[:max_seq_len]
            elif seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, dim, dtype=aa_repr.dtype)
                aa_repr = torch.cat([aa_repr, padding], dim=0)
        
        padded_aa_reprs.append(aa_repr)
    
    batch_aa = torch.stack(padded_aa_reprs, dim=0)
    
    return batch_aa, list(keys), list(media_mds)

def inference(args, clip_model, decoder_model, media_loader, logger):
    """
    按 batch=1 推理，打印结果和 media_md 对比，pause 等待用户输入继续
    """
    clip_model.eval()
    decoder_model.eval()
    
    logger.info(f"开始推理，共 {len(media_loader)} 个样本")
    
    with torch.no_grad():
        for idx, (batch_aa, keys, media_mds) in enumerate(media_loader):
            batch_aa = batch_aa.to(args.device)
            
            # 1. Extract Microbe Features
            padding_mask = (batch_aa.abs().sum(dim=-1) > 0)
            aa_embedding, _ = clip_model._forward_aa(batch_aa, padding_mask=padding_mask)
            input_vectors = aa_embedding  # [B, Dim]
            
            # 2. Generate Text
            generated_ids = decoder_model.generate(
                input_vectors=input_vectors,
                max_new_tokens=512,
                do_sample=False,  # 使用贪心解码
                temperature=1.0,
                top_p=0.9,
                pad_token_id=decoder_model.tokenizer.pad_token_id,
                eos_token_id=decoder_model.tokenizer.eos_token_id
            )
            
            # 3. Decode Generated Text
            # 只解码新生成的部分（去掉 prompt 部分）
            generated_text = decoder_model.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=False
            )
            
            # 提取 assistant 回复部分
            if "<|im_start|>assistant\n" in generated_text:
                predicted_text = generated_text.split("<|im_start|>assistant\n")[-1]
                # 移除可能的结束标记
                predicted_text = predicted_text.replace("<|im_end|>", "").strip()
            else:
                predicted_text = generated_text
            
            # 4. Print Results
            print("\n" + "="*80)
            print(f"样本 {idx + 1}/{len(media_loader)}")
            print(f"Key: {keys[0]}")
            print("-"*80)
            print("【预测结果】:")
            print(predicted_text)
            print("-"*80)
            print("【真实结果】:")
            print(media_mds[0])
            print("="*80)
            
            logger.info(f"样本 {idx + 1}/{len(media_loader)} (Key: {keys[0]}) 推理完成")
            
            # 5. Pause 等待用户输入
            user_input = input("\n按 Enter 继续下一个样本，输入 'q' 退出: ")
            if user_input.lower() == 'q':
                logger.info("用户选择退出推理")
                break

def load_decoder_checkpoint(args, decoder_model, logger):
    """
    加载 decoder 的 checkpoint
    """
    if not hasattr(args, 'decoder_checkpoint') or not args.decoder_checkpoint:
        logger.warning("未提供 decoder checkpoint，使用随机初始化的权重")
        return

    checkpoint_path = args.decoder_checkpoint
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"Checkpoint 文件不存在: {checkpoint_path}，使用随机初始化的权重")
        return

    logger.info(f"加载 Decoder checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        decoder_model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"从 checkpoint 加载 state_dict (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        decoder_model.load_state_dict(checkpoint, strict=False)
        logger.info("从 checkpoint 加载 state_dict")

def main():
    args = train_args()
    logger = setup_logger(args)
    logger.info("初始化 Media Inference...")
    logger.info(f"Device: {args.device}")

    # 模型加载参考 media_finetune.py 的 load_networks 函数
    clip_model, decoder_model = load_networks(args, logger)

    load_decoder_checkpoint(args, decoder_model, logger)

    # 加载推理数据集
    logger.info("加载推理数据集...")
    dataset = MediaDataset(args)
    logger.info(f"推理数据集大小: {len(dataset)}")

    # 创建 DataLoader，batch_size=1
    collate_fn = partial(collate_fn_media_inference, tokenizer=decoder_model.tokenizer)
    media_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False
    )
    
    # 执行推理
    logger.info("开始推理...")
    inference(args, clip_model, decoder_model, media_loader, logger)
    
    logger.info("推理完成")
