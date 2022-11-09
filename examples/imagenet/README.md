# EasyTorch Example - ImageNet Classification

Reference from [https://github.com/pytorch/examples/tree/main/imagenet](https://github.com/pytorch/examples/tree/main/imagenet)

## Train

### Resnet50 (One node)

```shell
easytrain -c configs/resnet50_8x_cfg.py
```

### MobileNet V3 Large (One node)

```shell
easytrain -c configs/mobilenet_v3_large_8x_cfg.py
```

### Resnet50 (Muti node)

Modify `CFG.DIST_INIT_METHOD='tcp://{ip_of_node_0}:{free_port}'` in `configs/resnet50_16x_cfg.py`.

e.g.

```python
CFG.DIST_INIT_METHOD='tcp://192.168.1.2:55555'
```

* Node 0:

```shell
easytrain -c configs/resnet50_16x_cfg.py
```

* Node 1:

```shell
easytrain -c configs/resnet50_16x_cfg.py --node-rank 1
```

### Other models

To train other models or modify hyperparameters, customize config yourself.

## Validate

### Resnet50

```shell
# last
python validate.py -c configs/resnet50_8x_cfg.py --devices 0

# best
python validate.py -c configs/resnet50_8x_cfg.py --devices 0 --ckpt /path/to/ckpt_dir/resnet50_best_val_acc@1.pt
```

### MobileNet V3 Large

```shell
python validate.py -c configs/mobilenet_v3_large_8x_cfg.py --devices 0
```
