from .registry import LOSS_REGISTRY


@LOSS_REGISTRY.register(name='A_LOSS')
class ALoss:
    def __init__(self, param_1, param_2) -> None:
        print('Init ALoss, param_1: {}, param_2: {}'.format(param_1, param_2))
