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

    # 1. Tokenizer
    property_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen 默认通常没有 pad_token，建议手动指定 eos 作为 pad
    if property_tokenizer.pad_token is None:
        property_tokenizer.pad_token = pad_token
        property_tokenizer.pad_token_id = property_tokenizer.convert_tokens_to_ids(pad_token)

    # 2. 回退到 AutoModelForCausalLM (为了解决报错)
    # 我们会在 forward 时通过 output_hidden_states=True 来获取表征
    property_encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if use_lora else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True if use_lora else False
    )

    if not use_lora:
        property_encoder = property_encoder.to(device)

    # 3. LoRA 配置
    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["c_attn", "c_proj", "w1", "w2"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="lora_only",
            # 【注意】虽然我们做特征提取，但因为基础模型是 CausalLM，
            # 设为 CAUSAL_LM 可以避免 PEFT 报 mismatch 错误。
            # 我们只需要在训练逻辑中忽略 lm_head 的输出即可。
            task_type=TaskType.CAUSAL_LM, 
        )

        property_encoder = get_peft_model(property_encoder, lora_config)

        # --- 关键：解决不收敛与显存问题 ---
        use_gradient_checkpointing = True

        if use_gradient_checkpointing:
            if hasattr(property_encoder, 'gradient_checkpointing_enable'):
                property_encoder.gradient_checkpointing_enable()

                # 【核心】必须禁用 KV Cache，否则梯度断裂
                property_encoder.config.use_cache = False
                print("已启用梯度检查点 (use_cache=False)")

        if hasattr(property_encoder, 'enable_input_require_grads'):
            property_encoder.enable_input_require_grads()

        print("=" * 50)
        print("Qwen CLIP Encoder Ready (Backbone only)")
        property_encoder.print_trainable_parameters()
        print("=" * 50)

    return property_encoder, property_tokenizer