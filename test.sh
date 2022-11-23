#!/bin/bash
#SBATCH -o test-1.out
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH -J test_bert

CUDA_VISIBLE_DEVICES=2 \
python test.py --model_path save_models/epoch_89.pth