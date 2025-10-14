import argparse


def data_provider_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    return args
