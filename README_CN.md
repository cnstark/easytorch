# EasyTorch

[![LICENSE](https://img.shields.io/github/license/cnstark/easytorch.svg)](https://github.com/cnstark/easytorch/blob/master/LICENSE)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/cnstark/easytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cnstark/easytorch/context:python)
[![gitee mirror](https://github.com/cnstark/easytorch/actions/workflows/git-mirror.yml/badge.svg)](https://gitee.com/cnstark/easytorch)

[English](README.md) **|** [简体中文](README_CN.md)

---

EasyTorch是一个基于PyTorch的开源神经网络框架，封装了PyTorch项目中常用的功能，帮助用户快速构建深度学习项目。

## :sparkles: 功能亮点

* **最小代码量**。EasyTorch封装了通用神经网络训练流程，用户仅需实现`Dataset`、`Model`以及训练/推理等关键代码，就能完成深度学习项目的构建。
* **万物基于Config**。用户通过配置文件控制训练模式与超参。EasyTorch根据配置文件内容的MD5自动生成唯一的结果存放目录，调整超参不再凌乱。
* **支持所有设备**。EasyTorch支持CPU、GPU与GPU分布式训练（单机多卡和多机多卡）。用户可以通过配置参数使用，不需要修改任何代码。
* **持久化训练日志**。支持`logging`日志系统与`Tensorboard`，并封装为统一接口，用户通过调用简单的接口即可保存自定义的训练日志。

## :wrench: 环境依赖

### 操作系统

* [Linux](https://pytorch.org/get-started/locally/#linux-prerequisites)
* [Windows](https://pytorch.org/get-started/locally/#windows-prerequisites)
* [MacOS](https://pytorch.org/get-started/locally/#mac-prerequisites)

推荐使用Ubuntu16.04及更高版本的系统。

### Python

python >= 3.6 （推荐 >= 3.9）

推荐使用[Miniconda](https://docs.conda.io/en/latest/miniconda.html)或者[Anaconda](https://www.anaconda.com/)

### PyTorch及CUDA

[pytorch](https://pytorch.org/) >= 1.4（推荐 >= 1.9）。
如需使用CUDA，请安装对应 CUDA 版本编译的 PyTorch 包。

注意：如需使用安培（Ampere）架构GPU，PyTorch版本需 >= 1.7 且CUDA版本 >= 11.0。

## :dart: 开始使用

### 安装EasyTorch

```shell
pip install easy-torch
```

### 初始化项目

TODO

## :pushpin: 示例

* [线性回归](examples/linear_regression)
* [MNIST手写数字识别](examples/mnist)
* [ImageNet图像分类](examples/imagenet)

*更多示例正在开发途中*

推荐参考成熟的开源项目[BasicTS](https://github.com/zezhishao/BasicTS)。

## :rocket: 引用

### BibTex 引用

如果EasyTorch对你的科研或工作有所帮助，可以考虑引用EasyTorch。
BibTex引用条目如下（需要`url`包）。

``` latex
@misc{wang2020easytorch,
  author =       {Yuhao Wang},
  title =        {{EasyTorch}: Simple and powerful pytorch framework.},
  howpublished = {\url{https://github.com/cnstark/easytorch}},
  year =         {2020}
}
```

### README 徽章

如果你的项目正在使用EasyTorch，可以将EasyTorch徽章 [![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch) 添加到你的 README 中：

```
[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
```

***（完整的文档即将推出）***
