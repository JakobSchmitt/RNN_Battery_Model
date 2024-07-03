#!/bin/bash

#SBATCH --job-name=alpha-train         # Name of the job
#SBATCH --nodes=1                      # Number of nodes to use
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPU to use
#SBATCH --mem-per-cpu=8G               # Memory per CPU core
#SBATCH --time=08:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --output=alpha-train.out       # output file
#SBATCH --error=alpha-train.err        # error file

# Load necessary modules
module load release/23.04 GCCcore/10.2.0 Python/3.8.6  
module load cuDNN/8.6.0.163-CUDA-11.8.0

# Activate the virtual environment
source /beegfs/ws/1/mamo674b-Thesis/venv/rnn-alpha/bin/activate

# Execute a Python script
python scripts/train.py lr=0.0001 "logger.tags=[stage_10, gru]" method.decoder_layers=1 method.encoder_layers=1 num_workers=4

