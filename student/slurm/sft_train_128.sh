#!/bin/bash
#SBATCH --job-name=sft_128
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=./logs/sft_%j.out
#SBATCH --error=./logs/sft_%j.err

module purge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/$USER/hf_cache

cd /scratch/$USER/nyu-llm-reasoners-a3

uv run python student/sft_train.py \
    --num-examples 128 \
    --lr 2e-5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --n-epochs 3 \
    --eval-every 20 \
    --max-eval-examples 200 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gpu-memory-utilization 0.6