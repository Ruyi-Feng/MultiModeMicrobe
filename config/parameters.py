import argparse


def data_provider_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='esm2_t30_150M_UR50D')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cut_off', type=int, default=512)
    parser.add_argument('--collective', action='store_true', default=False)
    args = parser.parse_args()
    return args

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args
