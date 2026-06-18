#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from collections.abc import Sequence

import torch


def sample_group_target_budgets(
    expanded_batch_size: int,
    num_generations: int,
    budget_choices: Sequence[int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample one target budget per GRPO prompt group.

    The expanded batch is expected to be laid out as contiguous groups of
    ``num_generations`` rollouts per prompt. If that invariant is not true, the
    function falls back to per-row sampling rather than silently reshaping.
    """
    if expanded_batch_size <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if not budget_choices:
        raise ValueError("budget_choices must be non-empty")

    choices = torch.tensor(list(budget_choices), device=device, dtype=torch.long)
    if expanded_batch_size % num_generations != 0:
        choice_idx = torch.randint(
            low=0,
            high=len(choices),
            size=(expanded_batch_size,),
            device=device,
        )
        return choices[choice_idx]

    n_groups = expanded_batch_size // num_generations
    group_idx = torch.randint(
        low=0,
        high=len(choices),
        size=(n_groups,),
        device=device,
    )
    return choices[group_idx].repeat_interleave(num_generations)


def assign_rollout_compute_styles(
    expanded_batch_size: int,
    num_generations: int,
    styles: Sequence[str] | None,
) -> list[str]:
    """Assign one compute style per expanded GRPO rollout row."""
    if expanded_batch_size <= 0:
        return []
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    style_list = list(styles or ["normal"])
    if not style_list:
        style_list = ["normal"]

    assigned = []
    for row_idx in range(expanded_batch_size):
        rollout_idx = row_idx % num_generations
        assigned.append(str(style_list[rollout_idx % len(style_list)]))
    return assigned
