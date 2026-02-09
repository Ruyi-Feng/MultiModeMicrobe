import argparse
import yaml


def data_provider_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='esm2_t12_35M_UR50D')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cut_off', type=int, default=1024)
    parser.add_argument('--collective', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='opt3d', help="normal, pH, temperature, salinity, oxygen")
    parser.add_argument('--split_data', type=str, default='train', help="train, test")
    parser.add_argument('--property_file_name', type=str, default='property.bin')
    parser.add_argument('--index_file_name', type=str, default='index.txt')
    parser.add_argument('--protein_index_file_name', type=str, default='protein_index.json')
    args = parser.parse_args()
    return args

def train_args():
    parser = argparse.ArgumentParser()
    # overall train parameters
    parser.add_argument('--mark', type=str, default='not_set')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--index_path', type=str, default='./data/collective/index.txt')
    parser.add_argument('--protein_index_path', type=str, default='./data/individual/protein_index.json')
    parser.add_argument('--data_path', type=str, default='./data/collective/')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--collective', action='store_true', default=False)
    parser.add_argument('--print_freq', type=int, default=2)

    # data provider parameters
    parser.add_argument('--model_name', type=str, default='esm2_t12_35M_UR50D')
    parser.add_argument('--cut_off', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=16)

    # aa encoder parameters
    parser.add_argument('--freeze_aa_encoder', action='store_true', default=False)
    parser.add_argument('--aa_repr_dim', type=int, default=480)
    parser.add_argument('--aa_encoder_hidden_dim', type=int, default=None)
    parser.add_argument('--aa_encoder_num_heads', type=int, default=4)
    parser.add_argument('--aa_encoder_num_layers', type=int, default=3)
    parser.add_argument('--aa_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--aa_encoder_type', type=str, default="cross_attention_fusion")

    # property encoder parameters
    parser.add_argument('--use_llm_property', action='store_true', default=False)

    # --- llm property encoder settings ---
    parser.add_argument('--property_model_name', type=str, default="Qwen/Qwen-1_8B")
    parser.add_argument('--freeze_property_encoder', action='store_true', default=False)
    parser.add_argument('--property_model_path', type=str, default="Qwen/Qwen-1_8B")
    parser.add_argument('--property_attn', action='store_true', default=False)

    # --- numerical property encoder settings ---
    parser.add_argument('--property_dim', type=int, default=4)
    parser.add_argument('--property_hidden_dim', type=int, default=64)

    parser.add_argument('--use_ph', action='store_true', default=False)
    parser.add_argument('--use_temp', action='store_true', default=False)
    parser.add_argument('--use_nacl', action='store_true', default=False)
    parser.add_argument('--use_oxygen', action='store_true', default=False)

    # LoRA parameters for property encoder
    parser.add_argument('--use_lora', action='store_true', default=False, help='Use LoRA for property encoder')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank (r)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                        help='Target modules for LoRA (e.g., c_attn c_proj w1 w2). If None, uses default for Qwen model')
    parser.add_argument('--lora_lr_multiplier', type=float, default=10.0,
                        help='Learning rate multiplier for LoRA parameters (default: 10.0, meaning LoRA lr = base_lr × 10)')


    # backbone parameters
    parser.add_argument('--cross_hidden_size', type=int, default=64)

    # training stability parameters
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs for learning rate')

    # decoder parameters (LoRA参数加在这里，是否使用，使用后rank，alpha，dropout)
    parser.add_argument('--decoder_model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--freeze_llm_decoder', action='store_true', default=False)
    parser.add_argument('--use_lora_decoder', action='store_true', default=False)
    parser.add_argument('--lora_rank_decoder', type=int, default=16)
    parser.add_argument('--lora_alpha_decoder', type=int, default=32)
    parser.add_argument('--lora_dropout_decoder', type=float, default=0.05)
    parser.add_argument('--decoder_checkpoint', type=str, default=None)
    parser.add_argument('--recipe_loss_weight', type=float, default=0.05)

    parser.add_argument('--use_yml', type=str, default='./config/scripts/individual_local_test.yml')


    args = parser.parse_args()

    if args.use_yml is not None:
        with open(args.use_yml, 'r') as f:
            args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
    return args
