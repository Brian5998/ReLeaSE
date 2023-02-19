#!/bin/bash
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=23zhou@da.org
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -p scavenger-gpu --gres=gpu:1
#SBATCH --exclusive
python AzidoazideTailored.py