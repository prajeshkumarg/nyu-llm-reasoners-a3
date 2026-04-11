import torch
from typing import Callable, Literal
import re
import ast
import json

def _extract_final_expression(response: str) -> tuple[str | None, str | None]:
    """
    Extract answer text and best expression from response.
    Returns (answer_text, expression) or (None, None) if no answer tags.
    """
    # must have <answer> tags
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if tag_match is None:
        return None, None

    answer_text = tag_match.group(1).strip()
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]

    # scan lines in reverse for equation matching target pattern
    for line in reversed(lines):
        eq_match = re.match(
            r"(?:Step\s*\d+\s*:\s*)?(.*?)=\s*(-?\d+\.?\d*)\s*$", line
        )
        if eq_match:
            return answer_text, eq_match.group(1).strip()

    # fallback: strip any trailing "= x" from last line
    last = re.sub(r"=.*$", "", lines[-1] if lines else answer_text).strip()
    return answer_text, last or answer_text


def _safe_eval(expr: str) -> float | None:
    """Evaluate an arithmetic expression safely, returns None on failure."""
    cleaned = re.sub(r"[^0-9+\-*/().\s]", "", expr)
    if not cleaned.strip():
        return None
    try:
        return float(eval(cleaned, {"__builtins__": {}}, {}))
    except Exception:
        return None


def countdown_answer_reward_fn(response: str, ground_truth) -> dict[str, float]:
    """
    Reward function for Countdown arithmetic puzzles.
    ground_truth: target number as int, float, or string.

    Returns dict with format_reward, answer_reward, reward.
    reward = 1 iff response has <answer> tags AND expression evaluates to target.
    """
    # parse ground truth target robustly (handles numpy scalars too)
    try:
        target = float(int(ground_truth))
    except (TypeError, ValueError):
        try:
            target = float(ground_truth)
        except (TypeError, ValueError):
            return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    # extract answer block and expression
    answer_text, expr = _extract_final_expression(response)

    # no answer tags → format failure
    if answer_text is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    # has tags but expression fails to evaluate
    result = _safe_eval(expr) if expr else None
    if result is None:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    # check if result matches target
    is_correct    = abs(result - target) < 1e-6
    answer_reward = 1.0 if is_correct else 0.0

    return {
        "format_reward": 1.0,
        "answer_reward": answer_reward,
        "reward":        answer_reward,
    }
def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list,
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
