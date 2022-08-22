from .registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ModelA:
    def __init__(self, param_1, param_2) -> None:
        print('Init ModelA, param_1: {}, param_2: {}'.format(param_1, param_2))
