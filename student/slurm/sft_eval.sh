#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=./logs/sft_eval_%A_%a.out
#SBATCH --error=./logs/sft_eval_%A_%a.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

mkdir -p logs

MODELS=(
    "/scratch/pg2973/sft_model_512"
    "/scratch/pg2973/sft_model_1024"
)
NAMES=(
    "sft_512"
    "sft_1024"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

uv run python student/evaluate.py \
    --model $MODEL \
    --max-examples 500 \
    --intellect-path /scratch/pg2973/data-distrib/intellect_math/test \
    --gpu-memory-utilization 0.85 \
    --math-log-file logs/math_outputs_${NAME}.log \
    --intellect-log-file logs/intellect_outputs_${NAME}.log \
    2>&1 | tee logs/eval_${NAME}.log