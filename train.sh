#!/bin/bash
#SBATCH -o train-1.out
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH -J train_bert

CUDA_VISIBLE_DEVICES=2 \
python train.py --checkpoint save_models/epoch_19.pth