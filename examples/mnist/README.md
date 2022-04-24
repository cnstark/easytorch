# EasyTorch Example - MNIST Classification

## Train

* CPU

```shell
easytrain -c configs/mnist_cpu_cfg.py
```

* GPU (1x)

```shell
easytrain -c configs/mnist_1x_cfg.py --gpus 0
```

## Validate

* CPU

```shell
python validate.py -c configs/mnist_cpu_cfg.py
```

* GPU (1x)

```shell
python validate.py -c configs/mnist_1x_cfg.py --gpus 0
```
