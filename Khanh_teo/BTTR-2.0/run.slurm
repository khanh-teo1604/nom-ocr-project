#!/usr/bin/env bash
#SBATCH --job-name=bttr_ids_train
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=32:00:00
#SBATCH --output=/home/projectg7/logs/bttr/%x-%j.out
#SBATCH --error=/home/projectg7/logs/bttr/%x-%j.err

set -euo pipefail

BTTR_DIR="/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/BTTR-2.0"
cd "$BTTR_DIR"
mkdir -p logs checkpoints

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "=================================================="

source /home/projectg7/.venv/bin/activate

python train_fit.py \
    --dataset_type nom_crop \
    --data_dir "/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops/MTH1000" \
    --pretrained_ckpt "checkpoints/epoch=epoch=04-step=step=210275-val_ExpRate=val_ExpRate=0.7940.ckpt" \
    --gpus 1 \
    --batch_size 8 \
    --max_epochs 30 \
    --check_val_every_n_epoch 1 \
    --save_top_k 5 \
    --early_stopping_patience 8 \
    --num_workers 4 \
    --patience 3 \
    --learning_rate 0.1 \
    --checkpoint_dir checkpoints2

