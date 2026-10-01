The code in this lab provides one simple PyTorch implementation of Bi-LSTM+CRF.

For a more complete implementation, see: https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF/

## Setup

1. Install Anaconda

   <a href="https://zhuanlan.zhihu.com/p/75717350">Windows installation guide (in Chinese)</a>

   Official documentation: <a href="https://docs.continuum.io/anaconda/install/">anaconda install</a>

2. Create a virtual environment and install PyTorch
    ```shell
    # Create the virtual environment
    conda create -n nlplab python=3.7	# creates a virtual environment named nlplab

    # Virtual environment commands
    conda activate nlplab  # activate nlplab; the prompt prefix should change from (base) to (nlplab)
    conda deactivate       # leave the current virtual environment
    conda info -e          # list all virtual environments; * marks the current one

    # Install PyTorch 1.6.0 (CPU)
    # Note: activate the nlplab environment before installing
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda install pytorch==1.6.0 cpuonly
    ```
## Running

1. Configure PyCharm

   Install PyCharm and set it to use the Anaconda virtual environment (<a href="https://jingyan.baidu.com/article/f3e34a12e7b015f5eb653523.html">reference, in Chinese</a>)

2. Install other dependencies

   ```sh
   # Run inside the nlplab virtual environment
   pip install -r requirements.txt
   ```

3. Train

   ```shell
   # save/ contains a roughly trained model, so you can skip training and go straight to inference

   # Prepare data (run inside data/)
   python data_u.py
   # Train the model (run from the project root)
   # If a GPU environment is installed and configured, add --cuda to train on the GPU
   python run.py
   ```

4. Inference

   ```shell
   python infer.py
   ```
