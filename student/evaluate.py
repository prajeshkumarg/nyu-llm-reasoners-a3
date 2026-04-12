"""Minimal evaluation script for MATH and Intellect test sets."""

from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths, log_file=None):
    """Run evaluation and return accuracy.

    Changes from original:
    - Added `log_file` param: if given, writes each example's output + rewards to that file
      so we can inspect raw model generations for the writeup analysis.
    - Now tracks 3 reward categories instead of just summing correct:
        (1) format=1, answer=1  -> model got format right AND correct answer
        (2) format=1, answer=0  -> model used \\boxed{} but wrong answer
        (3) format=0, answer=0  -> model never produced \\boxed{} at all
    - Prints category counts after grading.
    """
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    # counters for the 3 categories from the assignment
    counts = {"fmt1_ans1": 0, "fmt1_ans0": 0, "fmt0_ans0": 0}

    # collect log lines if we need to write them out
    log_lines = []

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])

        fmt = reward["format_reward"]
        ans = reward["answer_reward"]

        # bucket into one of the 3 categories
        if fmt == 1 and ans == 1:
            counts["fmt1_ans1"] += 1
        elif fmt == 1 and ans == 0:
            counts["fmt1_ans0"] += 1
        else:
            counts["fmt0_ans0"] += 1

        # build a log entry for this example so we can read it later
        log_lines.append(
            f"=== Example {i} | format_reward={fmt} | answer_reward={ans} ===\n"
            f"GT : {ground_truths[i]}\n"
            f"OUT: {text}\n"
        )

    # write all entries to disk if a path was given
    if log_file:
        with open(log_file, "w") as f:
            f.write("\n".join(log_lines))
        print(f"Outputs written to {log_file}")

    # print the 3-way breakdown required by the assignment
    total = len(outputs)
    print(f"\nCategory breakdown (n={total}):")
    print(f"  (1) format=1, answer=1 (correct)        : {counts['fmt1_ans1']}")
    print(f"  (2) format=1, answer=0 (boxed but wrong) : {counts['fmt1_ans0']}")
    print(f"  (3) format=0, answer=0 (no \\boxed found) : {counts['fmt0_ans0']}")

    return counts["fmt1_ans1"] / total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math_train_dev_test/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
   
    parser.add_argument("--math-log-file", default="math_outputs.log")

    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts, log_file=args.math_log_file)
    print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")
    # pass log_file so every model output is saved for manual inspection
    acc = evaluate(llm, prompts, gts, log_file="math_outputs.log")
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
