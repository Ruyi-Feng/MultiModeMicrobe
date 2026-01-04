import os
import torch
from config.parameters import train_args
from train import load_model, load_train_data


def check_convergence(args, base_model, model, batch_data):
    base_model.eval()
    model.eval()
    base_model.to(args.device)
    model.to(args.device)
    batch_data = batch_data.to(args.device)

    with torch.no_grad():
        padding_mask = (batch_data.abs().sum(dim=-1) > 0)
        # 按照 media_finetune.py 的调用形式，_forward_aa 返回 (embedding, hidden_states)
        base_output, _ = base_model._forward_aa(batch_data, padding_mask=padding_mask)
        output, _ = model._forward_aa(batch_data, padding_mask=padding_mask)

        cos_similarity = torch.cosine_similarity(base_output, output, dim=1)
        return cos_similarity.mean()


def main():
    args = train_args()

    # 确保有 base_checkpoint 参数，如果没有则需要手动指定或在 args 中添加
    if not hasattr(args, 'base_checkpoint'):
        # 默认值或报错，这里假设用户会在运行前设置好或 args 里有
        # 为了演示，可以设为 None 或抛出提示
        pass

    print("Loading Data...")
    train_loader = load_train_data(args)

    # 获取一个 batch 的数据
    batch = next(iter(train_loader))
    if args.collective:
        # collective mode: batch_p, batch_a, batch_keys
        batch_data = batch[1]
    else:
        # individual mode: batch_descriptions, batch_aa, padding_mask, batch_keys
        batch_data = batch[1]

    print(f"Data loaded. Shape: {batch_data.shape}")

    print("Loading Models...")
    # 加载 base model
    base_model, _ = load_model(args)

    base_model_path = "./checkpoints/checkpoint.pth.tar"
    checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
    base_model.load_state_dict(checkpoint["state_dict"])

    # 加载用于对比的 model 结构
    model, _ = load_model(args)

    target_checkpoint_paths = [
        {"epoch 5": "./checkpoints/checkpoint_5.pth.tar"},
        {"epoch 10": "./checkpoints/checkpoint_10.pth.tar"},
        {"epoch 15": "./checkpoints/checkpoint_15.pth.tar"},
        {"epoch 20": "./checkpoints/archaea_checkpoint_20.pth.tar"},
    ]

    print("Checking Convergence...")
    for name, path in target_checkpoint_paths.items():
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            continue

        print(f"Processing {name}: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            cos_sim = check_convergence(args, base_model, model, batch_data)
            print(f"Cos similarity ({name}): {cos_sim.item():.4f}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

