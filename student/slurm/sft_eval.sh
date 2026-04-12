#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=./logs/sft_eval_%A_%a.out
#SBATCH --error=./logs/sft_eval_%A_%a.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

MODELS=(
    "/scratch/pg2973/sft_model_128"
    "/scratch/pg2973/sft_model_full"
)
NAMES=(
    "sft_128"
    "sft_full"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

uv run python student/evaluate.py \
    --model $MODEL \
    --max-examples 500 \
    --intellect-path /scratch/pg2973/data-distrib/intellect_math/test \
    --gpu-memory-utilization 0.85 \
    --math-log-file logs/math_outputs_${NAME}.log \
    2>&1 | tee logs/eval_${NAME}.log