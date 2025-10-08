import time
from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(args, model, device, loader, data_config, optimizer):
    """
    这里接收的都是定义好的model, device, loader, optimizer
    其中loader是对比学习的loader,已经直接加载了对比学习的mini batch的
    每个loader传递的数据都已经加过了cls token, cls token的位置包含在data_config里
    经过了tokenizer
    """
    print("starting training")
    start_time = time.time()

    if model is not None:
        model.to(device)
        model.train()


    for step, batch_data in tqdm(enumerate(loader)):
        # 使用dataloader获取aa和property pairs
        aa_seq_batch, property_seq_batch = batch_data
        # aa B, S
        # property item B, S, H

        aa_seq_batch = aa_seq_batch.to(device)
        property_seq_batch = {k: v.to(device) for k, v in property_seq_batch.items()}

        optimizer.zero_grad()

        pred = model(aa_seq_batch, property_seq_batch)

        logits_aa = pred["logits_aa"]
        logits_property = pred["logits_property"]

        N = logits_aa.shape[0]
        labels = torch.arange(N).to(device)
        loss_a = F.cross_entropy(logits_aa, labels)
        loss_p = F.cross_entropy(logits_property, labels)
        loss = (loss_a + loss_p) / 2


