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
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the required libraries
```angular2html
pip install -r requirements.txt
```

## Update
### 2025/03/17
After a discussion with Batter, two modifications to be made:
1. After the specialist exam, there should be a preliminary diagnosis. This is editable, so doctors can modify if it is not accurate
2. Based on the premilinary diagnosis, there should be a treatment plan. Then after auxiliary exam, there should be final diagnosis appeared in the medical record.
