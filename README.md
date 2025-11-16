# Transformer情感分析项目

这是一个基于Transformer模型的情感分析项目，使用IMDB电影评论数据集进行训练和测试。

## 项目简介

实现了一个Transformer模型，用于对IMDB电影评论进行情感分析（正面/负面评价），模型效果不好，目的只是用来学习整个流程和熟悉Transformer框架。

## 文件结构

- [main.py](file:///d:/test/main.py): 主程序，负责数据加载、模型训练和测试
- [model.py](file:///d:/test/model.py): Transformer模型的实现
- [utils.py](file:///d:/test/utils.py): 数据处理工具函数
- [requirements.txt](file:///d:/test/requirements.txt): 项目依赖库列表

## 环境要求

- Python 3.7+
- PyTorch 2.0+
- 其他依赖详见[requirements.txt](file:///d:/test/requirements.txt)

## 安装步骤

1. 克隆或下载此项目到本地

2. 安装所需依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

直接运行主程序开始训练和测试模型：

```bash
python main.py
```

程序会自动完成以下步骤：
1. 下载IMDB数据集
2. 构建词汇表
3. 预处理数据
4. 初始化Transformer模型
5. 在训练集上训练模型
6. 在测试集上评估模型性能

## 模型参数

- 嵌入维度：256
- 批次大小：32
- 学习率：0.001
- 训练轮数：3

这些参数可以在[main.py](file:///d:/test/main.py)中修改。

## 注意事项

- 首次运行时会自动下载IMDB数据集，请确保网络连接正常
- 默认使用GPU训练，如果没有可用GPU会自动切换到CPU
- 可根据需要调整模型超参数以获得更好的性能
