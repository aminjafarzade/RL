#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from typing import Any
from typing import Iterable

import torch


BUDGET_FEATURE_NAMES = [
    "current_step_normalized",
    "nfe_used_normalized",
    "target_budget_normalized",
    "remaining_budget_normalized",
    "mask_ratio",
    "mean_confidence",
    "min_confidence",
    "max_confidence",
    "mean_entropy",
    "min_entropy",
    "max_entropy",
    "token_entropy",
    "is_masked",
    "normalized_position",
]

VERIFIER_FEATURE_NAMES = [
    "parse_ok",
    "format_ok",
    "arithmetic_unknown",
    "arithmetic_false",
    "arithmetic_true",
    "verifier_score",
    "has_answer_span",
    "remask_count_normalized",
    "answer_span_indicator",
]


def get_policy_extra_feature_names(config: Any) -> list[str]:
    names: list[str] = []
    if getattr(config, "enable_budget_conditioning", False):
        names.extend(BUDGET_FEATURE_NAMES)
    if getattr(config, "enable_verifier", False):
        names.extend(VERIFIER_FEATURE_NAMES)
    return names


def _safe_stat(values: torch.Tensor, mask: torch.Tensor, op: str) -> torch.Tensor:
    masked = values.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    if op == "mean":
        return masked.sum(dim=-1, keepdim=True) / denom
    if op == "min":
        return values.masked_fill(~mask, float("inf")).amin(dim=-1, keepdim=True)
    if op == "max":
        return values.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)
    raise ValueError(f"Unknown stat op {op}")


def _sanitize_stat(stat: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)


def _verifier_feature_value(name: str, verifier_state: dict[str, Any] | None) -> float:
    if verifier_state is None:
        if name == "arithmetic_unknown":
            return 1.0
        return 0.0
    if name == "parse_ok":
        return float(bool(verifier_state.get("parse_ok", False)))
    if name == "format_ok":
        return float(bool(verifier_state.get("format_ok", False)))
    arithmetic_ok = verifier_state.get("arithmetic_ok")
    if name == "arithmetic_unknown":
        return float(arithmetic_ok is None)
    if name == "arithmetic_false":
        return float(arithmetic_ok is False)
    if name == "arithmetic_true":
        return float(arithmetic_ok is True)
    if name == "verifier_score":
        return float(verifier_state.get("verifier_score", 0.0) or 0.0)
    if name == "has_answer_span":
        return float(bool(verifier_state.get("final_answer_span_text")))
    if name == "remask_count_normalized":
        return float(verifier_state.get("remask_count_normalized", 0.0) or 0.0)
    raise KeyError(name)


def build_policy_extra_features(
    feature_names: Iterable[str],
    mask_index: torch.Tensor,
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    steps_taken: torch.Tensor,
    max_steps: int,
    target_budget: int | torch.Tensor | None = None,
    verifier_states: list[dict[str, Any] | None] | None = None,
    remask_counts: torch.Tensor | None = None,
    answer_span_indicator: torch.Tensor | None = None,
) -> torch.Tensor | None:
    names = list(feature_names)
    if not names:
        return None

    B, L = mask_index.shape
    device = confidence.device
    dtype = confidence.dtype
    active_mask = torch.ones_like(mask_index, dtype=torch.bool)
    max_steps_float = max(float(max_steps), 1.0)

    if target_budget is None:
        target = torch.full((B, 1), max_steps_float, device=device, dtype=dtype)
    elif isinstance(target_budget, torch.Tensor):
        target = target_budget.to(device=device, dtype=dtype).view(B, 1)
    else:
        target = torch.full((B, 1), float(target_budget), device=device, dtype=dtype)
    safe_target = target.clamp_min(1.0)
    steps = steps_taken.to(device=device, dtype=dtype).view(B, 1)

    mean_conf = _sanitize_stat(_safe_stat(confidence, active_mask, "mean"))
    min_conf = _sanitize_stat(_safe_stat(confidence, active_mask, "min"))
    max_conf = _sanitize_stat(_safe_stat(confidence, active_mask, "max"))
    mean_entropy = _sanitize_stat(_safe_stat(entropy, active_mask, "mean"))
    min_entropy = _sanitize_stat(_safe_stat(entropy, active_mask, "min"))
    max_entropy = _sanitize_stat(_safe_stat(entropy, active_mask, "max"))
    mask_ratio = mask_index.float().mean(dim=-1, keepdim=True).to(dtype)
    positions = torch.linspace(0.0, 1.0, L, device=device, dtype=dtype).view(1, L)

    if remask_counts is None:
        remask_counts = torch.zeros(B, device=device, dtype=dtype)
    remask_counts = remask_counts.to(device=device, dtype=dtype).view(B, 1)

    if verifier_states is None:
        verifier_states = [None] * B
    if answer_span_indicator is None:
        answer_span_indicator = torch.zeros((B, L), device=device, dtype=dtype)
    else:
        answer_span_indicator = answer_span_indicator.to(device=device, dtype=dtype)

    columns = []
    for name in names:
        if name == "current_step_normalized":
            value = steps / max_steps_float
        elif name == "nfe_used_normalized":
            value = steps / max_steps_float
        elif name == "target_budget_normalized":
            value = target / max_steps_float
        elif name == "remaining_budget_normalized":
            value = (safe_target - steps).clamp_min(0.0) / safe_target
        elif name == "mask_ratio":
            value = mask_ratio
        elif name == "mean_confidence":
            value = mean_conf
        elif name == "min_confidence":
            value = min_conf
        elif name == "max_confidence":
            value = max_conf
        elif name == "mean_entropy":
            value = mean_entropy
        elif name == "min_entropy":
            value = min_entropy
        elif name == "max_entropy":
            value = max_entropy
        elif name == "token_entropy":
            columns.append(entropy.unsqueeze(-1))
            continue
        elif name == "is_masked":
            columns.append(mask_index.to(dtype).unsqueeze(-1))
            continue
        elif name == "normalized_position":
            columns.append(positions.expand(B, L).unsqueeze(-1))
            continue
        elif name == "answer_span_indicator":
            columns.append(answer_span_indicator.unsqueeze(-1))
            continue
        elif name in VERIFIER_FEATURE_NAMES:
            values = [
                _verifier_feature_value(
                    name,
                    {
                        **state,
                        "remask_count_normalized": float(
                            remask_counts[i].item() / max_steps_float
                        ),
                    }
                    if state is not None
                    else None,
                )
                for i, state in enumerate(verifier_states)
            ]
            value = torch.tensor(values, device=device, dtype=dtype).view(B, 1)
        else:
            raise ValueError(f"Unknown policy extra feature '{name}'")
        columns.append(value.expand(B, L).unsqueeze(-1))

    return torch.cat(columns, dim=-1)
