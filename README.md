# MegEngine RetinaNet for AI-Lesson

## 介绍

本目录包含了采用MegEngine实现的RetinaNet，并提供了在充气拱门数据集上的完整训练和测试代码

## 安装和环境配置

本目录下代码基于最新版MegEngine，在开始运行本目录下的代码之前，请确保已经正确安装MegEngine

## 如何使用

本目录提供了run.sh脚本，当准备工作完成之后（如数据、预训练模型等），可以一键跑通训练+测试+推理

- `git clone https://github.com/er-muyue/megengine-retinanet.git`

- 关于数据
  - 本目录使用的是充气拱门数据集，可以从这个位置下载：`s3://qlm-share/to/public/chongqigongmen`，放到当前目录的data文件夹下
  ```
  /path/to/
      |->chongqigongmen
      |    |images
      |    |train.odgt/json
      |    |test.odgt/json
  ```
  - tools/convert_odgt_to_coco.py 提供了将odgt格式转换为标准输入（coco json）的脚本
  - 目前该链接仅可内网访问，公共链接暂不开放

- 训练模型
  - `tools/train.py`的命令行选项如下：
    - `-f`, 所需要训练的网络结构描述文件
    - `-n`, 用于训练的devices(gpu)数量
    - `-w`, 预训练的backbone网络权重
    - `-b`，训练时采用的`batch size`, 默认2，表示每张卡训2张图
    - `-d`, 数据集的上级目录，默认`/data/datasets`
  - 默认情况下模型会存在 `log-of-模型名`目录下。

- 测试模型
  - `tools/test.py`的命令行选项如下：
    - `-f`, 所需要测试的网络结构描述文件
    - `-n`, 用于测试的devices(gpu)数量
    - `-w`, 需要测试的模型权重
    - `-d`，数据集的上级目录，默认`/data/datasets`
    - `-se`，连续测试的起始epoch数，默认为最后一个epoch，该参数的值必须大于等于0且小于模型的最大epoch数
    - `-ee`，连续测试的结束epoch数，默认等于`-se`（即只测试1个epoch），该参数的值必须大于等于`-se`且小于模型的最大epoch数

- 图片推理
  - `tools/inference.py`的命令行选项如下:
    - `-f`, 测试的网络结构描述文件。
    - `-w`, 需要测试的模型权重。
    - `-i`, 需要测试的样例图片。

- 一键运行
  - run.sh提供了一键运行脚本，默认用户已经申请了两块GPU
  
- 参考链接
  - 如遇问题，请参考 https://wiki.megvii-inc.com/pages/viewpage.action?pageId=248433046 进行修改
