import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum tensor elements (where mask==1) along dim, then divide by normalize_constant.

    Args:
        tensor: torch.Tensor
        mask: torch.Tensor, same shape as tensor; 1 = include, 0 = exclude.
        normalize_constant: float, divisor for normalization.
        dim: int | None, dimension to sum along; if None, sum over all elements.
    Returns:
        torch.Tensor, normalized masked sum.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # sum over sequence dim, normalize, then mean over batch
    loss = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant, dim=1
    ).mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss.detach(), {}

def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal LM.

    Args:
        model: HuggingFace causal LM.
        input_ids: torch.Tensor of shape (batch_size, sequence_length)
        labels: torch.Tensor of shape (batch_size, sequence_length), shifted input_ids.
        return_token_entropy: if True, also return per-token entropy.
    Returns:
        dict with "log_probs" (batch_size, sequence_length),
        and optionally "token_entropy" (batch_size, sequence_length).
    """
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)
    # Gather log-prob of each label token
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_tokens = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    combined = [p + o for p, o in zip(prompt_tokens, output_tokens)]
    max_len = max(len(seq) for seq in combined)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0


    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in combined]
    combined_tensor = torch.tensor(padded, dtype=torch.long)

    input_ids = combined_tensor[:, :-1]
    labels    = combined_tensor[:, 1:]

    batch_size = len(combined)
    seq_len    = max_len - 1
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for i, (p_tokens, o_tokens) in enumerate(zip(prompt_tokens, output_tokens)):
        L = len(p_tokens) + len(o_tokens)
        resp_start = len(p_tokens) - 1
        resp_end   = L - 1
        if resp_start < seq_len:
            response_mask[i, resp_start:resp_end] = 1

    return {
        "input_ids":     input_ids,
        "labels":        labels,
        "response_mask": response_mask,
    }