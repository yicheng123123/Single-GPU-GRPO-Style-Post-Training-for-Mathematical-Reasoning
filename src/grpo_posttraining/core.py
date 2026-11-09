"""Core functions preserved from the experimental adapters.

Bodies are unchanged; see provenance/core_function_map.json. Course interfaces
and student/project implementation are distinguished in docs/provenance.md.
The GRPO-Clip helper is retained, but was NOT enabled in the reported run.
"""
from __future__ import annotations
from typing import Callable, Literal, TYPE_CHECKING
import torch
from torch import Tensor
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    all_inputs = []
    all_labels = []
    all_masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt,add_special_tokens=False )
        output_ids = tokenizer.encode(output,add_special_tokens=False )
    
        full_ids=prompt_ids + output_ids
        input_ids=full_ids
        labels=full_ids[1:]
        mask=([0]*(len(prompt_ids)-1)+[1]*len(output_ids))
        all_inputs.append(input_ids)
        all_labels.append(labels)
        all_masks.append(mask)
    max_len = max(len(ids) for ids in all_inputs)
    pad_id=tokenizer.pad_token_id
    for i in range(len(all_inputs)):
        pad_count = max_len - len(all_inputs[i])

        all_inputs[i] += [pad_id] * pad_count
        all_labels[i] += [pad_id] * pad_count
        all_masks[i] += [0] * pad_count
    return {
        "input_ids": torch.tensor([id[:-1] for id in all_inputs], dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "response_mask": torch.tensor(all_masks, dtype=torch.bool),
    }

def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    
        reward_results = [
            reward_fn(response, ground_truth)
            for response, ground_truth in zip(
                rollout_responses, repeated_ground_truths
            )
        ]

        raw_rewards = torch.tensor(
            [result["reward"] for result in reward_results],
            dtype=torch.float32,
        )

        grouped_rewards = raw_rewards.reshape(-1, group_size)

        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        advantages = grouped_rewards - group_means

        if normalize_by_std:
            group_stds = grouped_rewards.std(dim=1, keepdim=True)
            advantages = advantages / (group_stds + advantage_eps)

        advantages = advantages.reshape(-1)

        metadata = {
            "mean_reward": raw_rewards.mean().item(),
        }

        return advantages, raw_rewards, metadata

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": token_log_probs}
    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)
        result["token_entropy"] = token_entropy
    return result

def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs

def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_loss = advantages * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = advantages * clipped_ratio
    loss = -torch.min(unclipped_loss, clipped_loss)
    clip_mask = unclipped_loss > clipped_loss

    return loss, {"clip_fraction": clip_mask.float().mean()}

def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        )
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        )
        metadata = {}

    elif loss_type == "grpo_clip":
        loss, metadata = run_compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss, metadata

def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_sum = (tensor * mask).sum(dim=dim)

    valid_count = mask.sum(dim=dim)

    

    return masked_sum / valid_count

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    normalize_prediction = run_masked_normalize(policy_log_probs, response_mask, dim=1, normalize_constant=normalize_constant)

    loss = -normalize_prediction.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss.detach()}

def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    per_response_loss = run_masked_mean(loss, response_mask, dim=1)
    loss = per_response_loss.mean() / gradient_accumulation_steps

  
    loss.backward()
    return loss.detach(), metadata

def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_masked_tensor = masked_tensor.sum(dim=dim)
    result= sum_masked_tensor / normalize_constant
    return result
