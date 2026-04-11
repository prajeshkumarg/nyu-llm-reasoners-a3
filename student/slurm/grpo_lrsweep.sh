#!/bin/bash
#SBATCH --job-name=grpo_lr_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=./logs/grpo_lr_%A_%a.out
#SBATCH --error=./logs/grpo_lr_%A_%a.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

LRS=(1e-6 3e-5 5e-5)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

uv run python student/grpo_train.py \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --train-path /scratch/pg2973/data-distrib/countdown/train_10k.parquet \
    --val-path   /scratch/pg2973/data-distrib/countdown/dev.parquet \
    --output-dir /scratch/pg2973/grpo_lr_${LR} \
    --n-grpo-steps 200 \
    --rollout-batch-size 16 \
    --group-size 8 \
    --sampling-temperature 0.7 \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --train-batch-size 16 \
    --grad-accum-steps 16 \
    --lr $LR \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-every 10 \
    --max-eval-examples 200 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gpu-memory-utilization 0.8