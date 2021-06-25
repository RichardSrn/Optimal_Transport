#! /bin/bash

#SBATCH -p GPU-DEPINFO
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o main.out
#SBATCH -e main.err
#SBATCH -J MAIN_ALGS


source /home_expes/tools/python/Python-3.8.7_gpu/bin/activate

srun --exclusive python3 ~/Optimal_Transport/deliverable_clean/test_algos/main.py
