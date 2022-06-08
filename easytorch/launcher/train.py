from typing import Dict


def train(cfg: Dict):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (Dict): Easytorch config.
    """

    # init runner
    runner = cfg['RUNNER'](cfg)

    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name='easytorch-training', log_file_name='training_log')

    # train
    runner.train(cfg)
