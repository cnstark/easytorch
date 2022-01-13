from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from easytorch.easyoptim.easy_lr_scheduler import MultiCosineAnnealingWarmupLR


if __name__ == '__main__':
    sw = SummaryWriter('logs')
    total_epoch = 100
    model = nn.Conv2d(3, 3, 3)
    sgd_opt = optim.SGD(model.parameters(), lr=1e-3)
    mc_lrs = MultiCosineAnnealingWarmupLR(sgd_opt, final_epoch=total_epoch, T_0=[total_epoch], lr_mult=[total_epoch], warmup_begin=10, warmup_factor=1e-3)

    lr_list = []
    for epoch_index in range(total_epoch):
        epoch = epoch_index + 1
        lr_list.append(mc_lrs.get_last_lr()[0])
        sw.add_scalar('lr', mc_lrs.get_last_lr()[0], global_step=epoch)
        mc_lrs.step()

    print(lr_list)
    sw.close()
    # plt.scatter([i for i in range(total_epoch)], lr_list)
    # plt.savefig('lr.png')
