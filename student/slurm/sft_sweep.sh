#!/bin/bash
#SBATCH --job-name=sft_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --output=./logs/sft_%A_%a.out
#SBATCH --error=./logs/sft_%A_%a.err

module purge
export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/nyu-llm-reasoners-a3

SIZES=(128 256 512 1024 0)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

if [ "$SIZE" -eq 0 ]; then
    NUM_EXAMPLES_ARG=""
else
    NUM_EXAMPLES_ARG="--num-examples $SIZE"
fi

uv run python student/sft_train.py \
    $NUM_EXAMPLES_ARG \
    --lr 2e-5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --n-epochs 3 \
    --eval-every 50 \
    --max-eval-examples 200 \
    --max-seq-len 512 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gpu-memory-utilization 0.6