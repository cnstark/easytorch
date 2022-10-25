import os
import sys
from argparse import ArgumentParser

from ..launcher import launch_training


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', help='training config', required=True)
    parser.add_argument('--node-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--devices', help='visible devices', type=str)
    return parser.parse_args()


def easytrain():
    # work dir
    path = os.getcwd()
    sys.path.append(path)

    # parse arguments
    args = parse_args()

    # train
    launch_training(args.cfg, args.devices, args.node_rank)
