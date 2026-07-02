# NLP Course Projects

本科自然语言处理课程实验项目。

## 项目概述

基于 PyTorch 实现 Bi-LSTM+CRF 模型，完成两个经典 NLP 序列标注任务：

| 实验 | 任务 | 目录 |
|------|------|------|
| 实验一 | 中文分词 (Chinese Word Segmentation) | `src/exp1/` |
| 实验二 | 命名实体识别 (Named Entity Recognition) | `src/exp2/` |

## 模型架构

```
Input → Embedding → Bi-LSTM → Linear → CRF → Output
```

两个任务均采用相同的 Bi-LSTM+CRF 结构，通过 CRF 层学习标签间的转移约束，提升序列标注效果。

## 目录结构

```
nlp/
├── 实验报告.pdf          # 课程实验报告
├── src/
│   ├── exp1/             # 实验一：中文分词
│   │   ├── model.py      # Bi-LSTM+CRF 模型定义
│   │   ├── dataloader.py # 数据加载与预处理
│   │   ├── run.py        # 训练脚本
│   │   ├── infer.py      # 推断脚本
│   │   ├── cws_result.txt# 分词结果输出
│   │   ├── data/         # 训练/测试数据
│   │   └── save/         # 模型保存目录
│   └── exp2/             # 实验二：命名实体识别
│       ├── model.py      # Bi-LSTM+CRF 模型定义
│       ├── dataloader.py # 数据加载与预处理
│       ├── run.py        # 训练脚本
│       ├── infer.py      # 推断脚本
│       ├── ner_result.txt# NER结果输出
│       ├── data/         # 训练/验证/测试数据
│       └── save/         # 模型保存目录
```

## 环境搭建

### 1. 安装 Anaconda

- [Windows 安装教程](https://zhuanlan.zhihu.com/p/75717350)
- [官方文档](https://docs.continuum.io/anaconda/install/)

### 2. 创建虚拟环境并安装 PyTorch

```shell
# 创建虚拟环境
conda create -n nlplab python=3.7

# 激活虚拟环境
conda activate nlplab

# 安装 PyTorch 1.6.0 CPU 版本
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch==1.6.0 cpuonly
```

### 3. 安装依赖

```shell
pip install -r src/exp1/requirements.txt
```

## 运行方式

### 实验一：中文分词

```shell
cd src/exp1

# 数据准备 (在 data 目录下运行)
cd data && python data_u.py && cd ..

# 训练 (可选 --cuda 使用 GPU)
python run.py

# 推断
python infer.py
```

### 实验二：命名实体识别

```shell
cd src/exp2

# 数据准备 (在 data 目录下运行)
cd data && python 0.split.py && python 1.data_u_ner.py && cd ..

# 训练 (可选 --cuda 使用 GPU)
python run.py

# 推断
python infer.py
```

## 参考

- [pytorch_NER_BiLSTM_CNN_CRF](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
