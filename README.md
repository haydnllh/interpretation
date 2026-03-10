# This is a library for interpreting black box models

Based on: https://christophm.github.io/interpretable-ml-book/

## Setting up the repository

Clone the repo
```
git clone https://github.com/haydnllh/interpretation.git
```

Set up conda environment:
```
conda create -n interpretation python=3.11
conda activate interpretation
```

Install requirements.txt
```
pip install -r requirements.txt
```

Install torch depending on your device \
For CPUs:
```
pip install torch
```
For CUDA:
```
pip install torch --index-url https://download.pytorch.org/whl/{YOUR_CUDA_VERSION}
```

To check CUDA version:
```
nvidia-smi
```