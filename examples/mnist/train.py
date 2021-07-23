import sys
sys.path.append('../..')
from argparse import ArgumentParser

from easytorch import launch_training


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', help='training config', required=True)
    parser.add_argument('--gpus', help='visible gpus', type=str)
    parser.add_argument('--tf32', help='enable tf32 on Ampere device', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus, args.tf32)
