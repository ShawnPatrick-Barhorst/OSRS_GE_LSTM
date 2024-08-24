#!/bin/bash

#SBATCH -J LSTM
#SBATCH -A AACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00

python osrs_torch.py
