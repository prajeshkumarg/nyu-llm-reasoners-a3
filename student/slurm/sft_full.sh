#!/bin/bash
#SBATCH --job-name=sft_full
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=./logs/sft_full_%j.out
#SBATCH --error=./logs/sft_full_%j.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

uv run python student/sft_train.py \
    --lr 2e-5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --n-epochs 1 \
    --eval-every 100 \
    --max-eval-examples 200 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gpu-memory-utilization 0.6