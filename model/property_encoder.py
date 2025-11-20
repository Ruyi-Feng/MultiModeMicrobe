
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def get_property_encoder(model_path="Qwen/Qwen-1_8B",
                         device="cuda",
                         pad_token="<|endoftext|>",
                         use_lora=False,
                         lora_r=64,
                         lora_alpha=16,
                         lora_dropout=0.05,
                         lora_target_modules=None):
    """
    加载 property encoder (Qwen 模型)

    Args:
        model_path: 模型路径
        device: 设备
        pad_token: padding token
        use_lora: 是否使用 LoRA
        lora_r: LoRA 的秩 (rank)
        lora_alpha: LoRA 的缩放因子
        lora_dropout: LoRA dropout 率
        lora_target_modules: LoRA 目标模块列表，如果为 None 则使用默认值

    Returns:
        property_encoder: 编码器模型
        property_tokenizer: tokenizer
    """
    property_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    property_tokenizer.pad_token = pad_token
    property_tokenizer.pad_token_id = property_tokenizer.convert_tokens_to_ids(pad_token)


    property_encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if use_lora else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True if use_lora else False
    )
    property_encoder.resize_token_embeddings(len(property_tokenizer))
    if not use_lora:
        property_encoder = property_encoder.to(device)

    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["c_attn", "c_proj", "w1", "w2"]

        lora_config = LoraConfig(
            r=lora_r,                          # LoRA 的秩
            lora_alpha=lora_alpha,             # LoRA 的缩放因子
            target_modules=lora_target_modules, # 目标模块
            lora_dropout=lora_dropout,         # LoRA dropout
            bias="none",                       # 不训练 bias
            task_type=TaskType.CAUSAL_LM,      # 任务类型
            modules_to_save=None               # 不保存额外模块
        )

        property_encoder = get_peft_model(property_encoder, lora_config)

        # 验证参数冻结情况
        trainable_params = sum(p.numel() for p in property_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in property_encoder.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # 启用梯度检查点以进一步减少显存占用
        if hasattr(property_encoder, 'gradient_checkpointing_enable'):
            property_encoder.gradient_checkpointing_enable()
            print("已启用梯度检查点以减少显存占用")

        # 确保输入需要梯度（用于梯度检查点）
        if hasattr(property_encoder, 'enable_input_require_grads'):
            property_encoder.enable_input_require_grads()

        print("=" * 50)
        print("LoRA 配置已应用，可训练参数信息：")
        property_encoder.print_trainable_parameters()
        print("=" * 50)

    return property_encoder, property_tokenizer
