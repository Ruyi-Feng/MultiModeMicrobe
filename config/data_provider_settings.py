import argparse


def data_provider_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='esmc_300m')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cut_off', type=int, default=64)

    args = parser.parse_args()
    return args
