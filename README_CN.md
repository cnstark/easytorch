# EasyTorch

[![LICENSE](https://img.shields.io/github/license/cnstark/easytorch.svg)](https://github.com/cnstark/easytorch/blob/master/LICENSE)

[English](README.md) **|** [简体中文](README_CN.md)

---

Easytorch是一个基于PyTorch的开源神经网络训练框架，封装了PyTorch项目中常用的功能，帮助用户快速构建深度学习项目。

## 功能亮点

* **最小代码量**。封装通用神经网络训练流程，用户仅需实现`Dataset`、`Model`以及训练/推理代码等关键代码，就能完成深度学习项目的构建。
* **万物基于Config**。通过配置文件控制训练模式与超参，根据配置内容的MD5自动生成唯一的结果存放目录，调整超参不再凌乱。
* **支持所有设备**。支持CPU、GPU与GPU分布式训练，通过配置参数一键完成设置。
* **持久化训练日志**。支持`logging`日志系统与`Tensorboard`，并封装为统一接口，用户通过一键调用即可保存自定义的训练日志。

## 环境依赖

### 操作系统

* [Linux](https://pytorch.org/get-started/locally/#linux-prerequisites)
* [Windows](https://pytorch.org/get-started/locally/#windows-prerequisites)
* [MacOS](https://pytorch.org/get-started/locally/#mac-prerequisites)

推荐使用Ubuntu16.04及更高版本或CentOS7及以更高版本。

### Python

python >= 3.6 （推荐 >= 3.7）

推荐使用[Anaconda](https://www.anaconda.com/)

### PyTorch及CUDA

[pytorch](https://pytorch.org/) >= 1.4（推荐 >= 1.7）

[CUDA](https://developer.nvidia.com/zh-cn/cuda-toolkit) >= 9.2 （推荐 >= 11.0）

注意：如需使用安培（Ampere）架构GPU，PyTorch版本需 >= 1.7且CUDA版本 >= 11.0。

### 其他依赖

```shell
pip install -r requirements.txt
```

## 示例

* [线性回归](examples/linear_regression)
* [MNIST手写数字识别](examples/mnist)

## README 徽章

如果你的项目正在使用EasyTorch，可以将EasyTorch徽章 [![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch) 添加到你的 README 中：

```
[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
```
