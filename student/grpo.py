import torch
from typing import Callable, Literal
import re

import re

def countdown_reward_fn(response: str, ground_truth) -> dict[str, float]:
    """
    Reward function for Countdown dataset.
    ground_truth: dict with keys 'numbers' and 'target'
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        import ast
        ground_truth = ast.literal_eval(ground_truth)

    target  = int(ground_truth["target"])
    numbers = [int(n) for n in ground_truth["numbers"]]

    # Step 1: Format check
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    format_reward = 1.0
    answer_text = match.group(1).strip()

    # Step 2: Extract expression
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
    expr_to_eval = None

    for line in reversed(lines):
        step_match = re.match(
            r"(?:Step\s*\d+\s*:\s*)?(.*?)=\s*(-?\d+\.?\d*)\s*$", line
        )
        if step_match:
            expr_to_eval = step_match.group(1).strip()
            expected_result = float(step_match.group(2))
            if abs(expected_result - target) < 1e-6:
                break
            expr_to_eval = None

    if expr_to_eval is None:
        single = re.sub(r"=.*$", "", answer_text.split("\n")[-1]).strip()
        expr_to_eval = single if single else answer_text

    # Step 3: Verify numbers used match provided numbers
    # For step format: collect all numbers from LEFT side of each equation
    # For single expression: just check that expression

    if len(lines) > 1:
        # Multi-step format: extract numbers from LEFT side of ALL steps
        all_expr_numbers = []
        for line in lines:
            step_match = re.match(r"(?:Step\s*\d+\s*:\s*)?(.*?)=\s*(-?\d+\.?\d*)\s*$", line)
            if step_match:
                left_side = step_match.group(1).strip()
                # only count numbers that appear as literals in left side
                # but exclude numbers that are results of previous steps
                nums_in_left = [int(n) for n in re.findall(r"\b\d+\b", left_side)]
                all_expr_numbers.extend(nums_in_left)
        
        # Filter: only check numbers that are in original list
        # intermediate results will fail the check → only keep original numbers
        available = numbers.copy()
        numbers_valid = True
        for n in all_expr_numbers:
            if n in available:
                available.remove(n)
            # intermediate results (e.g. 79) are allowed — skip if not in available
            # but if a number appears more than allowed times → invalid
        
        # Simpler check: at least all original numbers appear somewhere in answer
        # and no number outside original list is used more than possible
        used_from_original = []
        for n in all_expr_numbers:
            if n in numbers:
                used_from_original.append(n)
        
        # Check no original number used more times than available
        available = numbers.copy()
        for n in used_from_original:
            if n in available:
                available.remove(n)
            else:
                numbers_valid = False
                break
    else:
        # Single expression
        expr_numbers = [int(n) for n in re.findall(r"\b\d+\b", expr_to_eval or answer_text)]
        available = numbers.copy()
        numbers_valid = True
        for n in expr_numbers:
            if n in available:
                available.remove(n)
            else:
                numbers_valid = False
                break

    if not numbers_valid:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    # Step 4: Evaluate expression
    try:
        safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expr_to_eval)
        if not safe_expr.strip():
            return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
        result = eval(safe_expr, {"__builtins__": {}}, {})
        answer_reward = 1.0 if abs(float(result) - target) < 1e-6 else 0.0
    except Exception:
        answer_reward = 0.0

    return {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "reward":        float(format_reward * answer_reward),
    }
def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size % group_size == 0
    n_groups = rollout_batch_size // group_size

    raw_rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, ground_truth)
        raw_rewards.append(result["reward"])
        format_rewards.append(result["format_reward"])
        answer_rewards.append(result["answer_reward"])

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)

    rewards_grouped = raw_rewards_tensor.view(n_groups, group_size)

    group_means = rewards_grouped.mean(dim=1, keepdim=True)

    group_stds = rewards_grouped.std(dim=1, keepdim=True)

    advantages_grouped = rewards_grouped - group_means

    if normalize_by_std:
        advantages_grouped = advantages_grouped / (group_stds + advantage_eps)

    advantages = advantages_grouped.view(rollout_batch_size)

    metadata = {
        "mean_reward":         raw_rewards_tensor.mean().item(),
        "std_reward":          raw_rewards_tensor.std().item(),
        "max_reward":          raw_rewards_tensor.max().item(),
        "min_reward":          raw_rewards_tensor.min().item(),
        "mean_format_reward":  sum(format_rewards) / len(format_rewards),
        "mean_answer_reward":  sum(answer_rewards) / len(answer_rewards),
        "frac_all_correct":    float((rewards_grouped.sum(dim=1) == group_size).float().mean().item()),
        "frac_all_wrong":      float((rewards_grouped.sum(dim=1) == 0).float().mean().item()),
    }

    return advantages, raw_rewards_tensor, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages

    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * advantages

    loss = -torch.min(unclipped, clipped)

    is_clipped = (clipped < unclipped)
    clip_fraction = is_clipped.float().mean()

    metadata = {
        "clip_fraction": clip_fraction,
        "is_clipped": is_clipped,
        "mean_ratio": ratio.mean(),
        "max_ratio": ratio.max(),
        "min_ratio": ratio.min(),
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, \
            "raw_rewards required for loss_type='no_baseline'"

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, \
            "advantages required for loss_type='reinforce_with_baseline'"

    elif loss_type == "grpo_clip":
        assert advantages is not None, \
            "advantages required for loss_type='grpo_clip'"
        assert old_log_probs is not None, \
            "old_log_probs required for loss_type='grpo_clip'"
        assert cliprange is not None, \
            "cliprange required for loss_type='grpo_clip'"

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Must be one of: 'no_baseline', 'reinforce_with_baseline', 'grpo_clip'"
        )

    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}

    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    
    if dim is None:
        total_sum   = (tensor * mask).sum()
        total_count = mask.sum().clamp(min=1)
        return total_sum / total_count

    else:
        sum_along_dim   = (tensor * mask).sum(dim=dim)
        count_along_dim = mask.sum(dim=dim).float()
        
        result = sum_along_dim / count_along_dim
        return result

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss_per_example = masked_mean(
        tensor=per_token_loss,
        mask=response_mask,
        dim=1,
    )

    loss = loss_per_example.mean()

    scaled_loss = loss / gradient_accumulation_steps

    scaled_loss.backward()

    metadata["loss"] = loss.detach()

    return scaled_loss.detach(), metadata
