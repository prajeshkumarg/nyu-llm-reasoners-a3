import argparse
import os
import torch
import wandb
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from student.sft import tokenize_prompt_and_output, get_response_log_probs
from student.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
    countdown_answer_reward_fn as countdown_reward_fn,  # alias keeps rest of code unchanged
)


def load_prompt(name: str = "countdown") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def init_vllm(model_id, device, seed, gpu_memory_utilization=0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def make_prompt(nums, target, prompt_template):
    question = (
        f"Using the numbers in the list {list(nums)}, "
        f"create an equation that equals {target}. "
        f"You can use basic arithmetic operations (+, -, *, /) "
        f"and each number can only be used once."
    )
    return prompt_template.replace("{question}", question)


def make_ground_truth(nums, target):
    return str({"numbers": list(nums), "target": int(target)})


def evaluate_on_countdown(llm, df_val, prompt_template, max_examples=200):
    df = df_val.head(max_examples)
    prompts, gts = [], []
    for _, row in df.iterrows():
        prompts.append(make_prompt(row["nums"], row["target"], prompt_template))
        gts.append(str(int(row["target"])))  

    params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
    outputs = llm.generate(prompts, params)

    correct = 0
    fmt_correct = 0
    ans_correct = 0
    for out, gt in zip(outputs, gts):
        text = out.outputs[0].text + "</answer>"
        reward = countdown_reward_fn(text, gt)
        correct += reward["reward"]
        fmt_correct += reward["format_reward"]
        ans_correct += reward["answer_reward"]

    n = len(outputs)
    return {
        "accuracy":       correct / n,
        "format_reward":  fmt_correct / n,
        "answer_reward":  ans_correct / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train-path", default="/scratch/pg2973/data-distrib/countdown/train_10k.parquet")
    parser.add_argument("--val-path",   default="/scratch/pg2973/data-distrib/countdown/dev.parquet")
    parser.add_argument("--output-dir", default="/scratch/pg2973/grpo_model")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device",   default="cuda:1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    # GRPO hyperparameters
    parser.add_argument("--n-grpo-steps",        type=int,   default=200)
    parser.add_argument("--rollout-batch-size",  type=int,   default=16)
    parser.add_argument("--group-size",          type=int,   default=8)
    parser.add_argument("--sampling-temperature",type=float, default=0.7)
    parser.add_argument("--sampling-max-tokens", type=int,   default=1024)
    parser.add_argument("--train-batch-size",    type=int,   default=64)
    parser.add_argument("--grad-accum-steps",    type=int,   default=128)
    parser.add_argument("--lr",                  type=float, default=1e-5)
    parser.add_argument("--advantage-eps",       type=float, default=1e-6)
    parser.add_argument("--loss-type",           type=str,   default="reinforce_with_baseline")
    parser.add_argument("--use-std-normalization", action="store_true", default=True)
    parser.add_argument("--cliprange",           type=float, default=0.2)
    parser.add_argument("--eval-every",          type=int,   default=10)
    parser.add_argument("--max-eval-examples",   type=int,   default=200)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    args = parser.parse_args()

    # ── sanity checks ──────────────────────────────────────────────────────
    assert args.train_batch_size % args.grad_accum_steps == 0
    micro_train_batch_size = args.train_batch_size // args.grad_accum_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size

    # ── wandb ──────────────────────────────────────────────────────────────
    wandb.init(
        project="llm-reasoners-a3-grpo",
        name=f"grpo_{args.loss_type}_lr{args.lr}",
        config=vars(args),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # ── model + tokenizer ──────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.policy_device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # ── vLLM ───────────────────────────────────────────────────────────────
    print("Loading vLLM...")
    llm = init_vllm(
        args.model, args.vllm_device, args.seed, args.gpu_memory_utilization
    )

    # ── data ───────────────────────────────────────────────────────────────
    prompt_template = load_prompt("countdown")
    df_train = pd.read_parquet(args.train_path).reset_index(drop=True)
    df_val   = pd.read_parquet(args.val_path).reset_index(drop=True)
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # ── training loop ──────────────────────────────────────────────────────
    train_step = 0
    eval_step  = 0

    for grpo_step in range(args.n_grpo_steps):

        # ── Line 3: sample questions ───────────────────────────────────────
        batch_df = df_train.sample(n=n_prompts_per_rollout_batch).reset_index(drop=True)

        # ── Line 4: set old policy = current policy ────────────────────────
        # (old_log_probs computed AFTER tokenization below)

        # ── Line 5: sample G outputs per question ──────────────────────────
        prompts = [
            make_prompt(row["nums"], row["target"], prompt_template)
            for _, row in batch_df.iterrows()
        ]
        # repeat each prompt G times
        repeated_prompts = [p for p in prompts for _ in range(args.group_size)]
        repeated_gts = [
            str(int(row["target"]))                      # ← new (just target number)
            for _, row in batch_df.iterrows()
            for _ in range(args.group_size)
        ]


        sampling_params = SamplingParams(
            temperature=args.sampling_temperature,
            min_tokens=args.sampling_min_tokens,   # ← add this
            max_tokens=args.sampling_max_tokens,
            stop=["</answer>"],
        )
        # sync latest weights into vLLM
        load_policy_into_vllm_instance(model, llm)
        rollout_outputs = llm.generate(repeated_prompts, sampling_params)
        rollout_responses = [
            o.outputs[0].text + "</answer>" for o in rollout_outputs
        ]

        # ── Line 6 & 7: compute rewards and advantages ─────────────────────
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=countdown_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        # advantages: (rollout_batch_size,) = (128,)

        # ── Tokenize all rollouts ──────────────────────────────────────────
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )
        T = 2048
        all_input_ids     = tokenized["input_ids"][:, :T]
        all_labels        = tokenized["labels"][:, :T]
        all_response_mask = tokenized["response_mask"][:, :T]

        # ── old log probs (before any gradient steps) ──────────────────────
        if args.loss_type == "grpo_clip":
            model.eval()
            with torch.inference_mode():
                all_old_log_probs = get_response_log_probs(
                    model,
                    all_input_ids.to(args.policy_device),
                    all_labels.to(args.policy_device),
                )["log_probs"].cpu()
            model.train()
        else:
            all_old_log_probs = None

        # ── Line 8-10: inner training loop ────────────────────────────────
        # advantages: (rollout_batch_size,) → reshape to (rollout_batch_size, 1)
        advantages_col  = advantages.unsqueeze(1)   # (128, 1)
        raw_rewards_col = raw_rewards.unsqueeze(1)  # (128, 1)

        optimizer.zero_grad()
        total_loss = 0.0

        for micro_start in range(0, args.rollout_batch_size, micro_train_batch_size):
            micro_end = micro_start + micro_train_batch_size

            m_input  = all_input_ids[micro_start:micro_end].to(args.policy_device)
            m_labels = all_labels[micro_start:micro_end].to(args.policy_device)
            m_mask   = all_response_mask[micro_start:micro_end].to(args.policy_device)
            m_adv    = advantages_col[micro_start:micro_end].to(args.policy_device)
            m_raw    = raw_rewards_col[micro_start:micro_end].to(args.policy_device)
            m_old    = (
                all_old_log_probs[micro_start:micro_end].to(args.policy_device)
                if all_old_log_probs is not None else None
            )

            log_probs_dict   = get_response_log_probs(model, m_input, m_labels)
            policy_log_probs = log_probs_dict["log_probs"]

            # token entropy for logging
            token_entropy = get_response_log_probs(
                model, m_input, m_labels, return_token_entropy=True
            )["token_entropy"]
            mean_entropy = masked_mean(token_entropy, m_mask).item()

            loss, meta = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=m_mask,
                gradient_accumulation_steps=args.grad_accum_steps,
                loss_type=args.loss_type,
                raw_rewards=m_raw,
                advantages=m_adv,
                old_log_probs=m_old,
                cliprange=args.cliprange,
            )
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # ── logging ────────────────────────────────────────────────────────
        log_dict = {
            "train/loss":          total_loss,
            "train/grad_norm":     grad_norm.item(),
            "train/reward":        reward_metadata["mean_reward"],
            "train/format_reward": reward_metadata["mean_format_reward"],
            "train/answer_reward": reward_metadata["mean_answer_reward"],
            "train/entropy":       mean_entropy,
            "train/frac_all_correct": reward_metadata["frac_all_correct"],
            "train/frac_all_wrong":   reward_metadata["frac_all_wrong"],
            "train_step": train_step,
        }
        if args.loss_type == "grpo_clip" and "clip_fraction" in meta:
            log_dict["train/clip_fraction"] = meta["clip_fraction"].item()

        wandb.log(log_dict)
        print(
            f"Step {grpo_step:3d} | "
            f"reward={reward_metadata['mean_reward']:.3f} | "
            f"fmt={reward_metadata['mean_format_reward']:.3f} | "
            f"ans={reward_metadata['mean_answer_reward']:.3f} | "
            f"loss={total_loss:.4f} | "
            f"grad_norm={grad_norm.item():.3f}"
        )
        train_step += 1

        # ── eval ───────────────────────────────────────────────────────────
        if (grpo_step + 1) % args.eval_every == 0:
            print(f"\nEvaluating at step {grpo_step+1}...")
            model.eval()
            load_policy_into_vllm_instance(model, llm)

            val_metrics = evaluate_on_countdown(
                llm, df_val, prompt_template, args.max_eval_examples
            )
            print(
                f"  Val reward={val_metrics['accuracy']:.4f} | "
                f"fmt={val_metrics['format_reward']:.4f} | "
                f"ans={val_metrics['answer_reward']:.4f}"
            )
            wandb.log({
                "eval/reward":        val_metrics["accuracy"],
                "eval/format_reward": val_metrics["format_reward"],
                "eval/answer_reward": val_metrics["answer_reward"],
                "eval_step": eval_step,
            })
            # ── log example rollouts for writeup ──────────────────────────
            print(f"\n  Example rollouts at step {grpo_step+1}:")
            for idx in range(min(3, len(rollout_responses))):
                reward = countdown_reward_fn(
                    rollout_responses[idx],
                    repeated_gts[idx]
                )
                print(f"  [{idx}] reward={reward['reward']} | {rollout_responses[idx][:200]}")

            eval_step += 1
            model.train()

    # ── save ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()