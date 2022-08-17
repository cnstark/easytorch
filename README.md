# EasyTorch

[![LICENSE](https://img.shields.io/github/license/cnstark/easytorch.svg)](https://github.com/cnstark/easytorch/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/easy-torch)](https://pypi.org/project/easy-torch/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/cnstark/easytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/cnstark/easytorch/context:python)
[![python lint](https://github.com/cnstark/easytorch/actions/workflows/pylint.yml/badge.svg)](https://github.com/cnstark/easytorch/blob/master/.github/workflows/pylint.yml)

[English](README.md) **|** [简体中文](README_CN.md)

EasyTorch is an open source neural network framework based on PyTorch, which encapsulates common functions in PyTorch projects to help users quickly build deep learning projects.

## :sparkles: Highlight Characteristics

* :computer: **Minimum Code**. EasyTorch encapsulates the general neural network training pipeline. Users only need to implement key codes such as `Dataset`, `Model`, and training/inference to build deep learning projects.
* :wrench: **Everything Based on Config**. Users control the training mode and hyperparameters through the config file. EasyTorch automatically generates a unique result storage directory according to the MD5 of the config file content, which help users to adjust hyperparameters more conveniently.
* :flashlight: **Support All Devices**. EasyTorch supports CPU, GPU and GPU distributed training (single node multiple GPUs and multiple nodes). Users can use it by setting parameters without modifying any code.
* :page_with_curl: **Save Training Log**. Support `logging` log system and `Tensorboard`, and encapsulate it as a unified interface, users can save customized training logs by calling simple interfaces.

## :cd: Dependence

### OS

* [Linux](https://pytorch.org/get-started/locally/#linux-prerequisites)
* [Windows](https://pytorch.org/get-started/locally/#windows-prerequisites)
* [MacOS](https://pytorch.org/get-started/locally/#mac-prerequisites)

Ubuntu 16.04 and later systems are recommended.

### Python

python >= 3.6 (recommended >= 3.9)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended.

### PyTorch and CUDA

[pytorch](https://pytorch.org/) >= 1.4 (recommended >= 1.9).
To use CUDA, please install the PyTorch package compiled with the corresponding CUDA version.

Note: To use Ampere GPU, PyTorch version >= 1.7 and CUDA version >= 11.0.

## :dart: Get Started

### Installation

```shell
pip install easy-torch
```

### Initialize Project

TODO

## :pushpin: Examples

* [Linear Regression](examples/linear_regression)
* [MNIST Digit Recognition](examples/mnist)
* [ImageNet Image Classification](examples/imagenet)

*More examples are on the way*

It is recommended to refer to the excellent open source project [BasicTS](https://github.com/zezhishao/BasicTS).

## :rocket: Citations

### BibTex Citations

If EasyTorch helps your research or work, please consider citing EasyTorch.
The BibTex reference item is as follows(requires the `url` LaTeX package).

``` latex
@misc{wang2020easytorch,
  author =       {Yuhao Wang},
  title =        {{EasyTorch}: Simple and powerful pytorch framework.},
  howpublished = {\url{https://github.com/cnstark/easytorch}},
  year =         {2020}
}
```

### README Badge

If your project is using EasyTorch, please consider put the EasyTorch badge [![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch) add to your README.

```
[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
```

***(Full documentation is coming soon)***
