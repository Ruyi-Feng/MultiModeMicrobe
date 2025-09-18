from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


class QwenModel(nn.Module):
    def __init__(self, model_name):
        super(QwenModel, self).__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, input_vectors):
        pred = self.qwen(input_vectors)
        decode = self.tokenizer.decode(pred[0], skip_special_tokens=True)
        return decode


