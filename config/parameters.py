import argparse


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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--index_path', type=str, default='./data/individual/index.txt')
    parser.add_argument('--protein_index_path', type=str, default='./data/individual/protein_index.json')
    parser.add_argument('--data_path', type=str, default='./data/individual/')
    parser.add_argument('--max_seq_len', type=int, default=16)

    parser.add_argument('--model_name', type=str, default='esm2_t12_35M_UR50D')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cut_off', type=int, default=512)
    parser.add_argument('--collective', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=5)

    args = parser.parse_args()
    return args
