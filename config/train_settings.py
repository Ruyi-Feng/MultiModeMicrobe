import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args
