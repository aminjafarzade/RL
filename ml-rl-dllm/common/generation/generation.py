#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from typing import Any
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import bernoulli_sample
from common.generation.sampling import dpls_sample
from common.policy_features import build_policy_extra_features
from common.remasking.span_finder import find_final_answer_span
from common.remasking.strategies import RemaskState
from common.remasking.strategies import VerifierAnswerRemask
from common.verifiers.math_verifier import MathVerifier
from common.verifiers.math_verifier import has_answer_marker


class GenerationResult(NamedTuple):
    sequences: torch.Tensor  # Generated sequences (B, prompt_L + gen_L)
    steps_taken: torch.Tensor  # Steps taken per batch item

    # Policy training data (None for non-policy modes)
    sampling_inputs: torch.Tensor | None = None  # (B, T, BL)
    samples: torch.Tensor | None = None  # (B, T, BL)
    sampling_masks: torch.Tensor | None = None  # (B, T, BL)
    policy_inputs: tuple[torch.Tensor, ...] | None = None
    still_masked: torch.Tensor | None = None  # (B,)
    metadata: list[dict[str, Any]] | None = None
    step_traces: list[list[dict[str, Any]]] | None = None


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _target_budget_limit(
    target_budget: int | torch.Tensor | None,
    max_steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | int:
    if target_budget is None:
        return max_steps
    if isinstance(target_budget, torch.Tensor):
        return target_budget.to(device=device).view(batch_size).clamp(max=max_steps)
    return min(max_steps, int(target_budget))


def _decode_generation(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        [int(x) for x in token_ids.detach().cpu().tolist()],
        skip_special_tokens=True,
    )


def _verifier_to_dict(verifier_result) -> dict[str, Any] | None:
    if verifier_result is None:
        return None
    if hasattr(verifier_result, "to_dict"):
        return verifier_result.to_dict()
    return dict(verifier_result)


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _should_run_verifier(
    schedule: str,
    current_step: int,
    num_block: int,
    num_blocks: int,
    text: str,
    every_k: int,
) -> bool:
    if schedule == "none":
        return False
    if schedule == "final_only":
        return False
    if schedule == "every_k_steps":
        return every_k > 0 and current_step % every_k == 0
    if schedule == "final_block_only":
        return num_block == num_blocks - 1
    if schedule == "after_answer_detected":
        return has_answer_marker(text)
    raise ValueError(f"Unknown verifier_schedule: {schedule}")


@torch.no_grad()
def generate_unified(
    model,
    prompt: torch.Tensor,
    remasking: str,
    policy=None,
    thres: float | torch.Tensor | None = None,
    steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    sampling_mode: str = "bernoulli",
    dpls_stop_logit: float = 0.0,
    model_type: str | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature_policy: float = 1.0,
    full_context: bool = False,
    confidences_top_p: int = 1,
    tokenizer=None,
    enable_verifier: bool = False,
    verifier_type: str = "math",
    verifier_schedule: str = "final_only",
    verifier_every_k_steps: int = 8,
    verifier_threshold: float = 0.7,
    enable_remasking: bool = False,
    remask_strategy: str = "none",
    max_remasks_per_sample: int = 1,
    remask_cooldown_steps: int = 2,
    min_steps_before_remask: int = 0,
    remask_answer_span_mode: str = "numeric_only",
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    log_step_traces: bool = False,
) -> GenerationResult:
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
    elif remasking == "fastdllm":
        if thres is None:
            raise ValueError("thres must be provided for remasking='fastdllm'")
    elif remasking in ["low_confidence", "random"]:
        if steps is None:
            raise ValueError(f"steps must be provided for remasking='{remasking}'")
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    B, prompt_L = prompt.shape
    L = gen_length
    x = torch.full((B, L + prompt_L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_L] = prompt
    steps_taken = torch.zeros((B,), dtype=torch.int32, device=x.device)
    num_blocks = L // block_length

    if attention_mask is not None:
        _attn_mask = torch.ones((B, L + prompt_L), dtype=torch.float, device=x.device)
        _attn_mask[:, :prompt_L] = attention_mask.float()
        if model_type == "Dream":
            _attn_mask = _attn_mask.unsqueeze(1).unsqueeze(-2) * _attn_mask.unsqueeze(
                1
            ).unsqueeze(-1)
        # Handle DDP-wrapped models
        model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
        _attn_mask = _attn_mask.to(model_dtype)
    else:
        _attn_mask = None

    # Strategy-specific state
    record_policy_data = policy is not None
    sampling_history = [] if record_policy_data else None
    policy_extra_feature_names = policy_extra_feature_names or []

    if enable_verifier and tokenizer is None:
        raise ValueError("tokenizer must be provided when enable_verifier=True")
    if enable_verifier and verifier_type != "math":
        raise ValueError(f"Unsupported verifier_type: {verifier_type}")
    verifier = MathVerifier() if enable_verifier else None
    online_verifier_schedule = verifier_schedule
    if (
        enable_verifier
        and enable_remasking
        and remask_strategy == "verifier_answer"
        and verifier_schedule == "final_only"
    ):
        online_verifier_schedule = "after_answer_detected"
    verifier_remask = (
        VerifierAnswerRemask(
            max_remasks_per_sample=max_remasks_per_sample,
            remask_cooldown_steps=remask_cooldown_steps,
            min_steps_before_remask=min_steps_before_remask,
            verifier_threshold=verifier_threshold,
        )
        if enable_remasking and remask_strategy == "verifier_answer"
        else None
    )
    remask_states = [RemaskState() for _ in range(B)]
    verifier_calls = [0 for _ in range(B)]
    remasked_token_count = [0 for _ in range(B)]
    generated_text_before_remask = [None for _ in range(B)]
    generated_text_after_remask = [None for _ in range(B)]
    last_verifier_results = [None for _ in range(B)]
    answer_span_records = [None for _ in range(B)]
    answer_span_indicator = torch.zeros((B, L), dtype=torch.float, device=x.device)
    last_confidence = torch.zeros((B, L), dtype=torch.float, device=x.device)
    last_entropy = torch.zeros((B, L), dtype=torch.float, device=x.device)
    step_traces = [[] for _ in range(B)] if log_step_traces else None

    max_steps = L
    if remasking in ["low_confidence", "random"]:
        assert steps is not None and steps <= L
        tokens_per_step = L // steps
        max_steps = steps
    step_limit = _target_budget_limit(target_budget, max_steps, B, x.device)

    policy_type = None
    if policy is not None:
        policy_type = (
            policy.module.policy_type
            if hasattr(policy, "module")
            else policy.policy_type
        )

    for num_block in range(num_blocks):
        start_idx = num_block * block_length
        end_idx = start_idx + block_length
        block_slice = slice(start_idx, end_idx)
        block_index = torch.zeros(L, dtype=torch.bool, device=x.device)
        block_index[start_idx:end_idx] = True

        for _ in range(block_length):
            generation_part = x[:, prompt_L:]
            active_budget = steps_taken < step_limit
            mask_index = (generation_part == mask_id) & active_budget.unsqueeze(-1)
            block_mask_index = mask_index[:, block_index]  # (B, BL)

            if (~block_mask_index).all():
                break

            model_output = model(
                x,
                attention_mask=_attn_mask,
                output_hidden_states=(policy_type == "dit_hidden"),
            )

            # Handle Dream model logit shifting
            # Dream: logits at position i predict token i+1
            # For generated tokens at [P, P+1, ..., P+L-1], we need logits at [P-1, P, ..., P+L-2]
            if model_type == "Dream":
                logits = model_output.logits[
                    :, prompt_L - 1 : -1
                ]  # Include last prompt pos, exclude last gen pos
            else:
                logits = model_output.logits[
                    :, prompt_L:
                ]  # Just slice to generation portion

            # Apply Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute softmax once (needed by all strategies)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(
                dim=-1
            )
            last_confidence = confidence.detach().float()
            last_entropy = entropy.detach().float()

            # Get unmask decisions based on strategy
            if remasking == "policy":
                unmask, sampling_data = _policy_unmask_decisions(
                    mask_index,
                    block_mask_index,
                    probs,
                    x0,
                    steps_taken,
                    block_slice,
                    L,
                    policy,
                    policy_type,
                    sampling_mode,
                    full_context,
                    confidences_top_p,
                    model_output,
                    prompt_L,
                    dpls_stop_logit,
                    temperature_policy,
                    target_budget,
                    policy_extra_feature_names,
                    entropy,
                    [_verifier_to_dict(result) for result in last_verifier_results],
                    torch.tensor(
                        [state.remask_count for state in remask_states],
                        device=x.device,
                    ),
                    answer_span_indicator,
                )
                sampling_history.append(sampling_data)

            elif remasking == "fastdllm":
                unmask = _confidence_threshold_unmask(
                    block_mask_index, probs, block_slice, thres
                )
                if policy is not None:
                    sampling_data = _record_policy_data(
                        mask_index,
                        block_mask_index,
                        probs,
                        steps_taken,
                        block_slice,
                        L,
                        policy,
                        policy_type,
                        full_context,
                        confidences_top_p,
                        model_output,
                        prompt_L,
                        temperature_policy,
                        unmask,
                        target_budget,
                        policy_extra_feature_names,
                        entropy,
                        [_verifier_to_dict(result) for result in last_verifier_results],
                        torch.tensor(
                            [state.remask_count for state in remask_states],
                            device=x.device,
                        ),
                        answer_span_indicator,
                    )
                    sampling_history.append(sampling_data)

            elif remasking in ["low_confidence", "random"]:
                unmask = _fixed_step_unmask_decisions(
                    block_mask_index,
                    probs,
                    x0,
                    block_slice,
                    tokens_per_step,
                    remasking,
                )

            # Apply unmasking
            x[:, prompt_L:] = torch.where(unmask, x0, generation_part)

            # Update steps taken: only count steps for batch elements that had work to do
            active_this_step = block_mask_index.any(dim=-1)
            current_steps = steps_taken + active_this_step.int()

            if verifier is not None:
                for batch_idx in range(B):
                    if not bool(active_this_step[batch_idx].item()):
                        continue
                    decoded = _decode_generation(tokenizer, x[batch_idx, prompt_L:])
                    current_step = int(current_steps[batch_idx].item())
                    if not _should_run_verifier(
                        online_verifier_schedule,
                        current_step,
                        num_block,
                        num_blocks,
                        decoded,
                        verifier_every_k_steps,
                    ):
                        continue
                    verifier_result = verifier.verify(decoded)
                    verifier_calls[batch_idx] += 1
                    last_verifier_results[batch_idx] = verifier_result
                    span = find_final_answer_span(
                        tokenizer=tokenizer,
                        decoded_text=decoded,
                        generated_token_ids=x[batch_idx, prompt_L:],
                        verifier_result=verifier_result,
                        confidences=confidence[batch_idx],
                        min_token_index=0,
                        special_token_ids={mask_id},
                        remask_span_mode=remask_answer_span_mode,
                    )
                    answer_span_records[batch_idx] = (
                        span.to_dict() if span is not None else None
                    )
                    answer_span_indicator[batch_idx].zero_()
                    if span is not None:
                        answer_span_indicator[batch_idx, span.token_indices] = 1.0

                    remask_decision = None
                    if verifier_remask is not None:
                        remask_decision = verifier_remask.select(
                            verifier_result=verifier_result,
                            span=span,
                            state=remask_states[batch_idx],
                            current_step=current_step,
                            prompt_token_count=0,
                        )
                        if remask_decision.should_remask:
                            if generated_text_before_remask[batch_idx] is None:
                                generated_text_before_remask[batch_idx] = decoded
                            local_indices = torch.tensor(
                                remask_decision.token_indices,
                                dtype=torch.long,
                                device=x.device,
                            )
                            x[batch_idx, prompt_L + local_indices] = mask_id
                            verifier_remask.record(
                                remask_states[batch_idx],
                                remask_decision,
                                current_step,
                            )
                            remasked_token_count[batch_idx] += len(
                                remask_decision.token_indices
                            )
                            generated_text_after_remask[batch_idx] = _decode_generation(
                                tokenizer, x[batch_idx, prompt_L:]
                            )

                    if step_traces is not None:
                        step_traces[batch_idx].append(
                            {
                                "step": current_step,
                                "block": num_block,
                                "verifier_score": verifier_result.verifier_score,
                                "parse_ok": verifier_result.parse_ok,
                                "format_ok": verifier_result.format_ok,
                                "arithmetic_ok": verifier_result.arithmetic_ok,
                                "answer_span_token_indices": span.token_indices
                                if span is not None
                                else [],
                                "remask_indices": remask_decision.token_indices
                                if remask_decision is not None
                                else [],
                                "remask_reason": remask_decision.reason
                                if remask_decision is not None
                                else "none",
                            }
                        )

            steps_taken = current_steps

    if verifier is not None and verifier_schedule != "none":
        for batch_idx in range(B):
            decoded = _decode_generation(tokenizer, x[batch_idx, prompt_L:])
            verifier_result = verifier.verify(decoded)
            verifier_calls[batch_idx] += 1
            last_verifier_results[batch_idx] = verifier_result
            span = find_final_answer_span(
                tokenizer=tokenizer,
                decoded_text=decoded,
                generated_token_ids=x[batch_idx, prompt_L:],
                verifier_result=verifier_result,
                confidences=last_confidence[batch_idx],
                min_token_index=0,
                special_token_ids={mask_id},
                remask_span_mode=remask_answer_span_mode,
            )
            answer_span_records[batch_idx] = span.to_dict() if span is not None else None
            answer_span_indicator[batch_idx].zero_()
            if span is not None:
                answer_span_indicator[batch_idx, span.token_indices] = 1.0

    if isinstance(step_limit, torch.Tensor):
        step_limit_values = step_limit.to(device=x.device)
    else:
        step_limit_values = torch.full((B,), int(step_limit), device=x.device)

    metadata = []
    for batch_idx in range(B):
        final_text = (
            _decode_generation(tokenizer, x[batch_idx, prompt_L:])
            if tokenizer is not None
            else None
        )
        conf_stats = _tensor_stats(last_confidence[batch_idx])
        ent_stats = _tensor_stats(last_entropy[batch_idx])
        verifier_dict = _verifier_to_dict(last_verifier_results[batch_idx])
        sample_target_budget = int(step_limit_values[batch_idx].item())
        sample_nfe = int(steps_taken[batch_idx].item())
        still_masked = bool((x[batch_idx, prompt_L:] == mask_id).any().item())
        if still_masked and sample_nfe >= sample_target_budget:
            stop_reason = "target_budget_exhausted"
        elif not still_masked:
            stop_reason = "all_unmasked"
        else:
            stop_reason = "max_steps_reached"
        metadata.append(
            {
                "generated_text_before_remask": generated_text_before_remask[
                    batch_idx
                ],
                "generated_text_after_remask": generated_text_after_remask[batch_idx],
                "generated_text_final": final_text,
                "nfe_used": sample_nfe,
                "actual_nfe": sample_nfe,
                "target_budget": sample_target_budget,
                "budget_error": sample_nfe - sample_target_budget,
                "verifier_calls": verifier_calls[batch_idx],
                "remask_count": remask_states[batch_idx].remask_count,
                "remasked_token_count": remasked_token_count[batch_idx],
                "final_verifier_score": verifier_dict["verifier_score"]
                if verifier_dict
                else None,
                "parse_ok": verifier_dict["parse_ok"] if verifier_dict else None,
                "format_ok": verifier_dict["format_ok"] if verifier_dict else None,
                "arithmetic_ok": verifier_dict["arithmetic_ok"]
                if verifier_dict
                else None,
                "parsed_answer": verifier_dict["final_answer"] if verifier_dict else None,
                "answer_span_text": verifier_dict["final_answer_span_text"]
                if verifier_dict
                else None,
                "answer_span_token_indices": answer_span_records[batch_idx][
                    "token_indices"
                ]
                if answer_span_records[batch_idx]
                else [],
                "answer_span_mean_confidence": answer_span_records[batch_idx][
                    "mean_confidence"
                ]
                if answer_span_records[batch_idx]
                else None,
                "answer_span_min_confidence": answer_span_records[batch_idx][
                    "min_confidence"
                ]
                if answer_span_records[batch_idx]
                else None,
                "mean_confidence": conf_stats["mean"],
                "min_confidence": conf_stats["min"],
                "entropy_stats": ent_stats,
                "stop_reason": stop_reason,
                "verifier_result": verifier_dict,
            }
        )

    # Prepare metadata for gradient steps/loss computation
    if record_policy_data:
        generation_part = x[:, prompt_L:]
        still_masked = (generation_part == mask_id).any(dim=-1)

        if sampling_history:
            # Stack all sampling data for training
            sampling_inputs = torch.stack(
                [h["sampling_inputs"] for h in sampling_history], dim=1
            )
            samples = torch.stack([h["samples"] for h in sampling_history], dim=1)
            sampling_masks = torch.stack(
                [h["sampling_masks"] for h in sampling_history], dim=1
            )

            # Stack policy inputs
            policy_input_columns = zip(*[h["policy_inputs"] for h in sampling_history])
            policy_inputs_result = tuple(
                torch.stack(col, dim=1) for col in policy_input_columns
            )
        else:
            sampling_inputs = samples = sampling_masks = None
            policy_inputs_result = None

        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            sampling_inputs=sampling_inputs,
            samples=samples,
            sampling_masks=sampling_masks,
            policy_inputs=policy_inputs_result,
            still_masked=still_masked,
            metadata=metadata,
            step_traces=step_traces,
        )
    else:
        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            metadata=metadata,
            step_traces=step_traces,
        )


def _get_masks(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    block_slice: slice,
    full_context: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_mask = mask_index if full_context else block_mask_index

    if full_context:
        # Policy sees full sequence (B, L), but we only sample in current block
        sampling_mask = torch.zeros_like(mask_index)
        sampling_mask[:, block_slice] = block_mask_index
    else:
        # Policy sees only block (B, BL), sample from same positions
        sampling_mask = policy_mask

    return policy_mask, sampling_mask


def _compute_policy_logits(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Compute policy logits and masks.

    :return: (policy_logits, policy_mask, sampling_mask, policy_inputs)
    """
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    policy_mask, sampling_mask = _get_masks(
        mask_index, block_mask_index, block_slice, full_context
    )

    topk_result = probs.topk(confidences_top_p, dim=-1)
    c_max_input = (
        topk_result.values if full_context else topk_result.values[:, block_slice]
    )
    if policy_extra_feature_names:
        confidence = probs.max(dim=-1).values
        if entropy is None:
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(
                dim=-1
            )
        extra_features = build_policy_extra_features(
            feature_names=policy_extra_feature_names,
            mask_index=mask_index,
            confidence=confidence,
            entropy=entropy,
            steps_taken=steps_taken,
            max_steps=L,
            target_budget=target_budget,
            verifier_states=verifier_states,
            remask_counts=remask_counts,
            answer_span_indicator=answer_span_indicator,
        )
        if extra_features is not None:
            extra_input = extra_features if full_context else extra_features[:, block_slice]
            c_max_input = torch.cat([c_max_input, extra_input.to(c_max_input.dtype)], dim=-1)

    if policy_type == "dit_hidden":
        hidden_states = model_output.hidden_states[-1]
        hidden_states_input = (
            hidden_states[:, prompt_L:, :]
            if full_context
            else hidden_states[
                :, prompt_L + block_slice.start : prompt_L + block_slice.stop, :
            ]
        )
        policy_inputs = (policy_mask, hidden_states_input, per_batch_timestep)
    elif policy_type == "dit_confidence":
        policy_inputs = (policy_mask, c_max_input, per_batch_timestep)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy_logits = policy(*policy_inputs)

    # Apply temperature scaling
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    return policy_logits, policy_mask, sampling_mask, policy_inputs


def _policy_unmask_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    sampling_mode: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    dpls_stop_logit: float = 0.0,
    temperature_policy: float = 1.0,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    policy_logits, _, sampling_mask, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
        target_budget,
        policy_extra_feature_names,
        entropy,
        verifier_states,
        remask_counts,
        answer_span_indicator,
    )

    # Sample based on mode (using sampling_mask which is gated to current block)
    if sampling_mode == "bernoulli":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        samples_for_loglik = b
    elif sampling_mode == "bernoulli-argmax":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        # For batch items where nothing was selected, force unmask at argmax
        no_selection = b.sum(dim=-1) == 0
        if no_selection.any():
            masked_logits = policy_logits.clone()
            masked_logits[~sampling_mask] = -torch.inf
            force_idx = torch.argmax(masked_logits, dim=-1)
            batch_indices = torch.arange(b.shape[0], device=b.device)[no_selection]
            b[batch_indices, force_idx[no_selection]] = True
        samples_for_loglik = b
    elif sampling_mode == "dpls":
        dpls_sequences, b = dpls_sample(
            utilities=policy_logits,
            stop_logit=dpls_stop_logit,
            mask_index=sampling_mask,
        )
        samples_for_loglik = dpls_sequences
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    # Convert to sequence-level (always gate to current block)
    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    if full_context:
        unmask[:, block_slice] = b[:, block_slice]
    else:
        unmask[:, block_slice] = b

    sampling_data = {
        "sampling_inputs": policy_logits.detach(),
        "samples": samples_for_loglik.detach(),
        "sampling_masks": sampling_mask.detach(),
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }

    return unmask, sampling_data


def _confidence_threshold_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float | torch.Tensor,
) -> torch.Tensor:
    confidence = probs.max(dim=-1).values

    # Only consider masked positions in current block
    confidence_masked = confidence[:, block_slice].clone()
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    has_valid = block_mask_index.any(dim=-1)
    needs_force = has_valid & ~unmask_local.any(dim=-1)
    if needs_force.any():
        force_idx = torch.argmax(confidence_masked, dim=-1)
        batch_indices = torch.arange(unmask_local.shape[0], device=probs.device)
        unmask_local[batch_indices[needs_force], force_idx[needs_force]] = True

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


def _record_policy_data(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    unmask: torch.Tensor,
    target_budget: int | torch.Tensor | None = None,
    policy_extra_feature_names: list[str] | None = None,
    entropy: torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> dict:
    policy_logits, policy_mask, _, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
        target_budget,
        policy_extra_feature_names,
        entropy,
        verifier_states,
        remask_counts,
        answer_span_indicator,
    )

    samples = unmask if full_context else unmask[:, block_slice]

    # ES (Expert Steering) special behavior: save policy_mask (not sampling_mask) so the
    # model learns to mimic the confidence thresholding in a block-agnostic way
    return {
        "sampling_inputs": policy_logits.detach().clone(),
        "samples": samples.detach().clone(),
        "sampling_masks": policy_mask.detach().clone(),
        "policy_inputs": tuple(
            pi.detach().clone() if isinstance(pi, torch.Tensor) else pi
            for pi in policy_inputs
        ),
    }


def _fixed_step_unmask_decisions(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    block_slice: slice,
    tokens_per_step: int,
    mode: str,
) -> torch.Tensor:
    B = block_mask_index.shape[0]
    block_size = block_slice.stop - block_slice.start

    if mode == "low_confidence":
        confidence_block = torch.gather(
            probs[:, block_slice], dim=-1, index=x0[:, block_slice].unsqueeze(-1)
        ).squeeze(-1)
    elif mode == "random":
        confidence_block = torch.rand((B, block_size), device=x0.device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    confidence_masked = torch.where(block_mask_index, confidence_block, -torch.inf)

    num_masked_block = block_mask_index.sum(dim=-1)
    k = torch.clamp(num_masked_block, max=tokens_per_step)
    max_k = k.max().item()

    if max_k == 0:
        unmask = torch.zeros(
            (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
        )
        return unmask

    _, topk_indices = torch.topk(confidence_masked, k=max_k, dim=-1)

    positions = torch.arange(max_k, device=x0.device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < k.unsqueeze(-1)

    unmask_local = torch.zeros_like(block_mask_index, dtype=torch.bool)
    unmask_local.scatter_(1, topk_indices, valid_mask)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask
