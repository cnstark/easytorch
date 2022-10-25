from argparse import ArgumentParser

from easytorch import launch_runner, Runner


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', help='training config', required=True)
    parser.add_argument('--ckpt', help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument('--device-type', help='device type', type=str, default='gpu')
    parser.add_argument('--devices', help='visible devices', type=str)
    return parser.parse_args()


def main(cfg: dict, runner: Runner, ckpt: str = None):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    runner.load_model(ckpt_path=ckpt)

    runner.validate(cfg)


if __name__ == '__main__':
    args = parse_args()
    launch_runner(args.cfg, main, (args.ckpt, ), device_type=args.device_type, devices=args.devices)
