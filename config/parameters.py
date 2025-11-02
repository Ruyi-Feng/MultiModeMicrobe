import argparse
import yaml


def data_provider_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='esm2_t12_35M_UR50D')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cut_off', type=int, default=512)
    parser.add_argument('--collective', action='store_true', default=False)
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
    parser.add_argument('--aa_encoder_num_heads', type=int, default=4)
    parser.add_argument('--aa_encoder_num_layers', type=int, default=3)
    parser.add_argument('--aa_encoder_dropout', type=float, default=0.1)

    # property encoder parameters
    parser.add_argument('--freeze_property_encoder', action='store_true', default=False)
    parser.add_argument('--property_model_path', type=str, default="Qwen/Qwen-1_8B")

    # backbone parameters
    parser.add_argument('--cross_hidden_size', type=int, default=64)

    # decoder parameters (暂时没有这一部分)
    parser.add_argument('--freeze_llm_decoder', action='store_true', default=False)

    parser.add_argument('--use_yml', type=str, default=None)

    args = parser.parse_args()

    if args.use_yml is not None:
        with open(args.use_yml, 'r') as f:
            args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
    return args
