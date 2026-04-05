import argparse
import os
import torch
import wandb
from unittest.mock import patch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from student.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from student.evaluate import load_prompt, evaluate


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


def get_prompt_and_response(example):
    messages = example["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"]
    prompt = system + "\n\n" + user
    return prompt, assistant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-path", default="/scratch/pg2973/data-distrib/intellect_math/train")
    parser.add_argument("--output-dir", default="/scratch/pg2973/sft_model")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Subset size: 128, 256, 512, 1024, or None for full")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    # ── wandb ──────────────────────────────────────────────────────────────
    run_name = f"sft_n{args.num_examples or 'full'}_lr{args.lr}"
    wandb.init(project="llm-reasoners-a3-sft", name=run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # ── model + tokenizer ──────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.policy_device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── vLLM for eval ──────────────────────────────────────────────────────
    print("Loading vLLM...")
    llm = init_vllm(args.model, args.vllm_device, args.seed, args.gpu_memory_utilization)
    prompt_template = load_prompt("intellect")

    # ── dataset ────────────────────────────────────────────────────────────
    dataset = load_from_disk(args.data_path)
    if args.num_examples is not None:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))
    print(f"Training on {len(dataset)} examples")

    # ── MATH eval set — load once ──────────────────────────────────────────
    print("Loading MATH test set...")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    math_ds = math_ds.select(range(min(args.max_eval_examples, len(math_ds))))
    math_prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    math_gts = [ex["answer"] for ex in math_ds]

    # ── training loop ──────────────────────────────────────────────────────
    train_step = 0
    eval_step = 0

    for epoch in range(args.n_epochs):
        indices = torch.randperm(len(dataset)).tolist()

        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start: batch_start + args.batch_size]
            batch = [dataset[i] for i in batch_indices]

            prompt_strs, output_strs = [], []
            for ex in batch:
                p, r = get_prompt_and_response(ex)
                prompt_strs.append(p)
                output_strs.append(r)
                
            

            tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            T = args.max_seq_len
            input_ids     = tokenized["input_ids"][:, :T].to(args.policy_device)
            labels        = tokenized["labels"][:, :T].to(args.policy_device)
            response_mask = tokenized["response_mask"][:, :T].to(args.policy_device)

            # microbatch loop (gradient accumulation)
            micro_size = max(1, len(batch) // args.grad_accum_steps)
            total_loss = 0.0
            optimizer.zero_grad()
            for micro_start in range(0, len(batch), micro_size):
                micro_end = micro_start + micro_size
                m_input  = input_ids[micro_start:micro_end]
                m_labels = labels[micro_start:micro_end]
                m_mask   = response_mask[micro_start:micro_end]

                log_probs_dict   = get_response_log_probs(model, m_input, m_labels)
                policy_log_probs = log_probs_dict["log_probs"]

                normalize_constant = m_mask.sum().item()

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=m_mask,
                    gradient_accumulation_steps=args.grad_accum_steps,
                    normalize_constant=normalize_constant,
                )
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"Step {train_step} | loss: {total_loss:.4f}")
            wandb.log({"train/loss": total_loss, "train_step": train_step})
            train_step += 1

            # ── eval ───────────────────────────────────────────────────────
            if train_step % args.eval_every == 0:
                print(f"Step {train_step}: evaluating on MATH...")
                model.eval()
                load_policy_into_vllm_instance(model, llm)

                val_acc = evaluate(llm, math_prompts, math_gts)
                print(f"  MATH accuracy: {val_acc:.4f}")
                wandb.log({"eval/math_accuracy": val_acc, "eval_step": eval_step})
                eval_step += 1
                model.train()

    # ── save ───────────────────────────────────────────────────────────────
    out = f"{args.output_dir}_{args.num_examples or 'full'}"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved to {out}")
    wandb.finish()


if __name__ == "__main__":
    main()