#!/bin/bash
#SBATCH --job-name=grpo_dryrun
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=./logs/grpo_dry_%j.out
#SBATCH --error=./logs/grpo_dry_%j.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

uv run python student/grpo_train.py \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --train-path /scratch/pg2973/data-distrib/countdown/train_10k.parquet \
    --val-path   /scratch/pg2973/data-distrib/countdown/dev.parquet \
    --output-dir /scratch/pg2973/grpo_model_test \
    --n-grpo-steps 5 \
    --rollout-batch-size 4 \
    --group-size 2 \
    --sampling-temperature 0.7 \
    --sampling-max-tokens 256 \
    --train-batch-size 4 \
    --grad-accum-steps 4 \
    --lr 1e-5 \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-every 3 \
    --max-eval-examples 20 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gpu-memory-utilization 0.7