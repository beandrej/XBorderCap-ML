#!/bin/bash
#SBATCH --job-name=LSTM_NORM_16
#SBATCH --output=LSTM_NORM_16.out
#SBATCH --error=LSTM_NORM_16.err
#SBATCH --partition=normal.24h
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=23:59:00
#SBATCH --mail-type=END,FAIL

echo "🔄 Job started on $(hostname) at $(date)"
echo "🧠 Using $SLURM_CPUS_ON_NODE CPU cores"

module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

python -u train_FBMC.py

echo "✅ Job finished at $(date)"