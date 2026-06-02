import math
from types import SimpleNamespace

import pytest

from common.rewards import additive_proposal_reward
from common.rewards import multiplicative_budget_reward
from common.rewards import select_reward_functions_from_config


def test_additive_proposal_reward_normalized_formula():
    reward = additive_proposal_reward(
        task_score=1.0,
        nfe_used=32,
        max_steps=128,
        lambda_compute=0.2,
        normalize_compute=True,
    )
    assert reward == pytest.approx(1.0 - 0.2 * (32 / 128))


def test_additive_proposal_reward_raw_formula():
    reward = additive_proposal_reward(
        task_score=1.0,
        nfe_used=32,
        max_steps=128,
        lambda_compute=0.01,
        normalize_compute=False,
    )
    assert reward == pytest.approx(1.0 - 0.01 * 32)


def test_additive_proposal_wrong_fast_is_less_penalized_than_wrong_slow():
    wrong_fast = additive_proposal_reward(0.0, 8, 128, 0.2, True)
    wrong_slow = additive_proposal_reward(0.0, 64, 128, 0.2, True)
    assert wrong_fast > wrong_slow


def test_multiplicative_budget_wrong_answers_get_zero_reward():
    assert multiplicative_budget_reward(False, 8, 16, 128) == 0.0
    assert multiplicative_budget_reward(False, 64, 16, 128) == 0.0


def test_multiplicative_budget_penalizes_over_target_budget():
    within = multiplicative_budget_reward(True, 8, 16, 128)
    over = multiplicative_budget_reward(True, 64, 16, 128)
    assert within > over


def test_multiplicative_budget_penalizes_remasking():
    no_remask = multiplicative_budget_reward(True, 16, 16, 128, remask_count=0)
    two_remasks = multiplicative_budget_reward(True, 16, 16, 128, remask_count=2)
    assert no_remask > two_remasks


def test_multiplicative_budget_larger_target_reduces_budget_penalty():
    small_budget = multiplicative_budget_reward(True, 64, 16, 128)
    large_budget = multiplicative_budget_reward(True, 64, 64, 128)
    assert large_budget > small_budget


def test_multiplicative_budget_is_finite_for_edge_cases():
    reward = multiplicative_budget_reward(True, 0, 1, 1)
    assert math.isfinite(reward)


def test_config_selects_additive_proposal_reward_variant():
    selection = select_reward_functions_from_config(
        SimpleNamespace(
            reward_type="additive_proposal",
            reward_lambda_compute=0.2,
            reward_normalize_compute=True,
        )
    )
    assert selection.reward_function_names == ["mixed_additive_proposal_reward_func"]
    assert selection.parameters["reward_lambda_compute"] == 0.2


def test_config_selects_multiplicative_budget_reward_variant():
    selection = select_reward_functions_from_config(
        SimpleNamespace(
            reward_type="multiplicative_budget",
            reward_beta=2.0,
            reward_mu=0.5,
            reward_rho=0.1,
        )
    )
    assert selection.reward_function_names == [
        "mixed_multiplicative_budget_reward_func"
    ]
    assert selection.parameters["reward_beta"] == 2.0
