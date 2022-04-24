from argparse import ArgumentParser

from easytorch import launch_runner, Runner


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', help='training config', required=True)
    parser.add_argument('--ckpt', help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument('--gpus', help='visible gpus', type=str)
    return parser.parse_args()


def main(cfg: dict, runner: Runner, ckpt: str = None):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    runner.load_model(ckpt_path=ckpt)

    runner.validate(cfg)


if __name__ == '__main__':
    args = parse_args()
    launch_runner(args.cfg, main, (args.ckpt, ), gpus=args.gpus)
