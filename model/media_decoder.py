import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class MediaDecoder(nn.Module):
    def __init__(self, 
                 qwen_model_path="Qwen/Qwen2.5-3B-Instruct", 
                 input_vector_dim=1280,
                 lora_rank=16,
                 lora_alpha=32,
                 lora_dropout=0.05):
        super().__init__()
        
        # 1. 加载 Tokenizer 和 Model
        # Trust_remote_code=True 对于 Qwen 是必要的
        print(f"Loading Qwen2.5 from {qwen_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_path, 
            trust_remote_code=True,
            padding_side='right'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载基础模型
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 3. LoRA 配置 (参考 property_encoder 的逻辑)
        # 启用梯度检查点以节省显存 (Gradient Checkpointing)
        # 必须禁用 KV Cache，否则在训练时会报错或梯度断裂
        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False 
        
        # 启用输入梯度 (对于 LoRA + Gradient Checkpointing 通常是必须的)
        if hasattr(self.llm, "enable_input_require_grads"):
             self.llm.enable_input_require_grads()
        
        # 2. 配置并应用 LoRA
        # Qwen2.5 的 target_modules 通常包括所有线性层
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

        # 3. 定义 Projector (Input Embed Dim -> LLM Hidden Dim)
        # 获取 LLM 的 hidden size (Qwen2.5-3B 通常是 2560)
        # 注意：经过 PEFT 包装后，配置通常还在 self.llm.config 或 self.llm.base_model.model.config
        self.qwen_dim = self.llm.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(input_vector_dim, self.qwen_dim * 2),
            nn.GELU(),
            nn.Linear(self.qwen_dim * 2, self.qwen_dim)
        ).to(dtype=torch.bfloat16) # 确保 projector 精度一致
        
        # 4. 定义 Prompt 模板部分
        # 结构: [System Prompt] [User Start] [Vector] [User End] [Assistant Start]
        self.system_prompt = "<|im_start|>system\nYou are a biochemical assistant. You decode protein-based embeddings (ESM-2) into precise microbial culture media recipes.<|im_end|>\n"
        self.user_prompt_start = "<|im_start|>user\n"
        self.user_prompt_end = "\nGenerate the medium recipe for this microbial profile.<|im_end|>\n<|im_start|>assistant\n"

    def forward(self, input_vectors, target_token_ids=None, attention_mask=None):
        """
        input_vectors: [batch_size, vector_dim] (你的预训练向量)
        target_token_ids: [batch_size, seq_len] (期望输出的文本 IDs，即 Recipe)
        """
        device = input_vectors.device
        batch_size = input_vectors.shape[0]

        # 1. 投影向量
        # projected_embeds: [batch_size, 1, qwen_dim]
        # 确保 projector 在正确的 device 上 (如果 model device_map="auto"，projector 需要手动管理或跟随 input)
        self.projector.to(device)

        # 确保 input_vectors 是 bfloat16 以匹配 projector
        if input_vectors.dtype != torch.bfloat16:
            input_vectors = input_vectors.to(dtype=torch.bfloat16)

        projected_embeds = self.projector(input_vectors).unsqueeze(1).to(dtype=self.llm.dtype)

        # 2. 构建 Prompt Embeddings
        # 获取 input embeddings 层
        input_embedding_layer = self.llm.get_input_embeddings()

        # Part A: System + User Start
        text_part_a = self.system_prompt + self.user_prompt_start
        tokens_a = self.tokenizer(text_part_a, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        embeds_a = input_embedding_layer(tokens_a).expand(batch_size, -1, -1)

        # Part B: User End + Assistant Start
        text_part_b = self.user_prompt_end
        tokens_b = self.tokenizer(text_part_b, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        embeds_b = input_embedding_layer(tokens_b).expand(batch_size, -1, -1)

        # 3. 拼接输入 Embeddings
        # 顺序: [System+UserStart] -> [Vector] -> [UserEnd+AssistantStart] -> [Target(if train)]

        if target_token_ids is not None:
            # Training Mode
            target_embeds = input_embedding_layer(target_token_ids)
            
            inputs_embeds = torch.cat([embeds_a, projected_embeds, embeds_b, target_embeds], dim=1)
            
            # 4. 构建 Labels
            # Prompt 和 Vector 部分的 Loss 不计算 (-100)
            len_a = tokens_a.shape[1]
            len_vec = projected_embeds.shape[1] # 1
            len_b = tokens_b.shape[1]
            len_target = target_token_ids.shape[1]
            
            # 创建 masking labels
            ignore_labels = torch.full((batch_size, len_a + len_vec + len_b), -100, dtype=torch.long, device=device)
            labels = torch.cat([ignore_labels, target_token_ids], dim=1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=True
            )
            return outputs
            
        else:
            # Inference Mode (仅返回 logits 或用于 debug)
            inputs_embeds = torch.cat([embeds_a, projected_embeds, embeds_b], dim=1)
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                return_dict=True
            )
            return outputs

    def generate(self, input_vectors, max_new_tokens=256, **kwargs):
        """
        用于推理生成的辅助函数
        """
        device = input_vectors.device
        batch_size = input_vectors.shape[0]
        self.projector.to(device)
        
        # 确保 input_vectors 是 bfloat16 以匹配 projector
        if input_vectors.dtype != torch.bfloat16:
            input_vectors = input_vectors.to(dtype=torch.bfloat16)

        # 准备 Embeddings
        projected_embeds = self.projector(input_vectors).unsqueeze(1).to(dtype=self.llm.dtype)
        input_embedding_layer = self.llm.get_input_embeddings()
        
        text_part_a = self.system_prompt + self.user_prompt_start
        tokens_a = self.tokenizer(text_part_a, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        embeds_a = input_embedding_layer(tokens_a).expand(batch_size, -1, -1)
        
        text_part_b = self.user_prompt_end
        tokens_b = self.tokenizer(text_part_b, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        embeds_b = input_embedding_layer(tokens_b).expand(batch_size, -1, -1)
        
        inputs_embeds = torch.cat([embeds_a, projected_embeds, embeds_b], dim=1)
        
        # 调用 generate
        return self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, **kwargs)
