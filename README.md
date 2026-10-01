# NLP Course Projects

Lab projects from my undergraduate Natural Language Processing course.

## Overview

A Bi-LSTM+CRF model implemented in PyTorch for two classic NLP sequence-labeling tasks:

| Lab | Task | Directory |
|------|------|------|
| Lab 1 | Chinese Word Segmentation (CWS) | `src/exp1/` |
| Lab 2 | Named Entity Recognition (NER) | `src/exp2/` |

## Model Architecture

```
Input → Embedding → Bi-LSTM → Linear → CRF → Output
```

Both tasks use the same Bi-LSTM+CRF architecture. The CRF layer learns transition constraints between labels to improve sequence-labeling accuracy.

## Project Structure

```
nlp/
├── 实验报告.pdf          # Course lab report (in Chinese)
├── src/
│   ├── exp1/             # Lab 1: Chinese word segmentation
│   │   ├── model.py      # Bi-LSTM+CRF model definition
│   │   ├── dataloader.py # Data loading and preprocessing
│   │   ├── run.py        # Training script
│   │   ├── infer.py      # Inference script
│   │   ├── cws_result.txt# Segmentation output
│   │   ├── data/         # Training/test data
│   │   └── save/         # Saved models
│   └── exp2/             # Lab 2: Named entity recognition
│       ├── model.py      # Bi-LSTM+CRF model definition
│       ├── dataloader.py # Data loading and preprocessing
│       ├── run.py        # Training script
│       ├── infer.py      # Inference script
│       ├── ner_result.txt# NER output
│       ├── data/         # Training/validation/test data
│       └── save/         # Saved models
```

## Setup

### 1. Install Anaconda

- [Windows installation guide (in Chinese)](https://zhuanlan.zhihu.com/p/75717350)
- [Official documentation](https://docs.continuum.io/anaconda/install/)

### 2. Create a Virtual Environment and Install PyTorch

```shell
# Create the virtual environment
conda create -n nlplab python=3.7

# Activate it
conda activate nlplab

# Install PyTorch 1.6.0 (CPU)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch==1.6.0 cpuonly
```

### 3. Install Dependencies

```shell
pip install -r src/exp1/requirements.txt
```

## Running

### Lab 1: Chinese Word Segmentation

```shell
cd src/exp1

# Prepare data (run inside data/)
cd data && python data_u.py && cd ..

# Train (add --cuda to use a GPU)
python run.py

# Inference
python infer.py
```

### Lab 2: Named Entity Recognition

```shell
cd src/exp2

# Prepare data (run inside data/)
cd data && python 0.split.py && python 1.data_u_ner.py && cd ..

# Train (add --cuda to use a GPU)
python run.py

# Inference
python infer.py
```

## References

- [pytorch_NER_BiLSTM_CNN_CRF](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF/)
- [PyTorch documentation](https://pytorch.org/docs/)
