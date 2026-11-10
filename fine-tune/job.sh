#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal                 # or "long" (up to 7 days, 1 running job)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100_3g.20gb:1    # a100 | a100_3g.20gb | a100_2g.10gb | a100_2g.10gb_mps
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x-%j.err
##SBATCH --mail-type=END,FAIL        # uncomment for email
##SBATCH --mail-user=you@example.com

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) started at $(date)"
nvidia-smi -L

# --- load your environment ---
# module load python/3.11
# source ~/venvs/myenv/bin/activate

# --- your work ---
uv run python finetune2.py

echo "Finished at $(date)"