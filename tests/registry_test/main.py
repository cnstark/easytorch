from losses import LOSS_REGISTRY
from models import MODEL_REGISTRY

model_a = MODEL_REGISTRY.build('ModelA', {'param_1': 'mp1', 'param_2': 'mp2'})
a_loss = LOSS_REGISTRY.build('A_LOSS', {'param_1': 'lp1', 'param_2': 'lp2'})
l1_loss = LOSS_REGISTRY.build('L1_LOSS')
print(l1_loss)
