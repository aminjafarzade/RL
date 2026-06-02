#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import math
from dataclasses import dataclass
from typing import Any


def additive_proposal_reward(
    task_score: float,
    nfe_used: int | float,
    max_steps: int | float | None = None,
    lambda_compute: float = 0.01,
    normalize_compute: bool = True,
) -> float:
    if normalize_compute:
        if max_steps is None or max_steps <= 0:
            raise ValueError("max_steps must be positive when normalize_compute=True")
        compute_cost = float(nfe_used) / float(max_steps)
    else:
        compute_cost = float(nfe_used)
    return float(task_score) - float(lambda_compute) * compute_cost


def multiplicative_budget_reward(
    exact_match: int | bool | float,
    nfe_used: int | float,
    target_budget: int | float,
    max_steps: int | float,
    remask_count: int | float = 0,
    beta: float = 2.0,
    mu: float = 0.5,
    rho: float = 0.1,
    eps: float = 1e-8,
) -> float:
    if float(exact_match) <= 0.0:
        return 0.0
    safe_target_budget = max(float(target_budget), eps)
    safe_max_steps = max(float(max_steps), eps)
    over_budget = max(0.0, float(nfe_used) - safe_target_budget)
    budget_penalty = math.exp(-float(beta) * over_budget / safe_target_budget)
    compute_penalty = math.exp(-float(mu) * float(nfe_used) / safe_max_steps)
    remask_penalty = math.exp(-float(rho) * float(remask_count))
    return float(exact_match) * budget_penalty * compute_penalty * remask_penalty


@dataclass(frozen=True)
class RewardSelection:
    reward_type: str
    reward_function_names: list[str]
    parameters: dict[str, Any]


def select_reward_functions_from_config(config: Any) -> RewardSelection:
    reward_type = getattr(config, "reward_type", "multiplicative_budget")
    if reward_type == "additive_proposal":
        return RewardSelection(
            reward_type=reward_type,
            reward_function_names=["mixed_additive_proposal_reward_func"],
            parameters={
                "reward_lambda_compute": getattr(
                    config, "reward_lambda_compute", 0.01
                ),
                "reward_normalize_compute": getattr(
                    config, "reward_normalize_compute", True
                ),
            },
        )
    if reward_type == "multiplicative_budget":
        return RewardSelection(
            reward_type=reward_type,
            reward_function_names=["mixed_multiplicative_budget_reward_func"],
            parameters={
                "reward_beta": getattr(config, "reward_beta", 2.0),
                "reward_mu": getattr(config, "reward_mu", 0.5),
                "reward_rho": getattr(config, "reward_rho", 0.1),
            },
        )
    raise ValueError(
        f"Unknown reward_type '{reward_type}'. "
        "Expected 'additive_proposal' or 'multiplicative_budget'."
    )
