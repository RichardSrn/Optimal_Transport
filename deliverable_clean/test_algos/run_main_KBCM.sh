#! /bin/bash

#SBATCH -p GPU-DEPINFO
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH -o KBCM.out
#SBATCH -e KBCM.err
#SBATCH -J KBCM


source /home_expes/tools/python/Python-3.8.7_gpu/bin/activate

srun --exclusive python3 ~/Optimal_Transport/deliverable_clean/test_algos/main.py 2
