#!/bin/bash
#SBATCH --job-name=masac_train
#SBATCH --output=logs/masac_%j.out
#SBATCH --error=logs/masac_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=24G
#SBATCH -p ug-gpu-small

# activate your venv (assumes a venv/ directory in project root)
source ../../project_virtual_envrionment/bin/activate

# ensure local package is on PYTHONPATH
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

# run training
python -u masac/masac.py
