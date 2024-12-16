# LLM services

## Environment
Create a conda environment
```
conda create -n llm_service python=3.9
```

Switch to the newly created environment
```
conda activate llm_service
```
Install the PyTorch according to the CUDA version. For example, 
```angular2html
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the required libraries
```angular2html
pip install -r requirements.txt
```

