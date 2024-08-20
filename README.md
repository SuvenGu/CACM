# CACM

Pytorch implementation of Climate-adaptive Emergency Crop Monitoring in Inaccessible Regions with Satellite Imagery

## Requirements
* Python 3.7.7, PyTorch 1.13.1, and more in `requirements.txt`

## Setup Environment
Setup conda environment and activate

```
conda env create -f environment.yml
conda activate py37
```
All experiments were executed on a NVIDIA GeForce RTX 3090.

## Train
```
python tools/train.py --cfg experiments/CACM_sgd_lr5e-3_wd1e-4_bs4096_x100.yaml --dataDir data
```

## Evaluate
```
python tools/valid.py --cfg experiments/CACM_sgd_lr5e-3_wd1e-4_bs4096_x100.yaml --testModel output/data/CACM_sgd_lr5e-3_wd1e-4_bs4096_x100/model_best.pth.tar --dataDir data
```
