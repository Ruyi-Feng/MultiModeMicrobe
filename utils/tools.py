


def get_optimizer_params(model, weight_decay):
    """
    自动划分需要 weight decay / 不需要 weight decay 的参数组。
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 冻结参数跳过

        # 判断逻辑：
        if len(param.shape) == 1 or name.endswith(".bias"):
            # 1维参数（比如 LayerNorm.weight 或 bias）
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def compute_topk_accuracy(logits, target, topk=(1,)):
    """Helper: computes top-k accuracy for one direction."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # shape [N, maxk]
    correct = pred.eq(target.view(-1, 1))  # [N, maxk]
    res = []
    for k in topk:
        correct_k = correct[:, :k].any(dim=1).float().sum().mul_(100.0 / logits.size(0))
        res.append(correct_k)
    return res
