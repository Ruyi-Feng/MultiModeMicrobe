
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_property_encoder(model_path="Qwen/Qwen-1_8B",
                         device="cuda",
                         pad_token="<|endoftext|>"):
    property_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    property_tokenizer.pad_token = pad_token
    property_tokenizer.pad_token_id = property_tokenizer.convert_tokens_to_ids(pad_token)
    property_encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    property_encoder.resize_token_embeddings(len(property_tokenizer))
    property_encoder = property_encoder.to(device)

    return property_encoder, property_tokenizer
