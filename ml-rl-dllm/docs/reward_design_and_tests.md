# Reward Design and Test Requirements

This project must implement and test **two** reward functions.

The proposal reward and the recommended training reward are both part of the implementation. The proposal reward is not discarded; it is included as a named reward variant for ablation/baseline/tests.

## Reward variant 1: `additive_proposal`

### Formula

The project proposal uses:

```text
R = TaskScore - lambda_compute * ComputeCost
```

For math exact-match:

```text
R = exact_match - lambda_compute * nfe_used
```

or normalized:

```text
R = exact_match - lambda_compute * (nfe_used / max_steps)
```

### Suggested function signature

```python
def additive_proposal_reward(
    task_score: float,
    nfe_used: int,
    max_steps: int | None = None,
    lambda_compute: float = 0.01,
    normalize_compute: bool = True,
) -> float:
    ...
```

### Intended use

Use this reward for:
- proposal fidelity,
- ablation,
- diagnostic tests,
- comparison against multiplicative reward.

Do not make this the default final RL training reward unless intentionally running the ablation.

### Known problem

This reward can make a wrong-but-fast sample look better than a wrong-but-slow sample. That behavior should be documented and test-covered as a diagnostic property, not treated as a bug in the formula.

Example with normalized compute:

```text
wrong_fast = 0 - lambda * (8 / 128)
wrong_slow = 0 - lambda * (64 / 128)
wrong_fast > wrong_slow
```

This is why the multiplicative reward should be the default for final training.

## Reward variant 2: `multiplicative_budget`

### Formula

Recommended default:

```text
R =
    exact_match
    * exp(-beta * max(0, nfe_used - target_budget) / target_budget)
    * exp(-mu * nfe_used / max_steps)
    * exp(-rho * remask_count)
```

Where:
- `exact_match` is 1 for correct final parsed answer, 0 otherwise.
- `nfe_used` is frozen dLLM forward passes.
- `target_budget` is sampled or configured.
- `max_steps` is the max denoising horizon.
- `remask_count` is number of remask actions.
- `beta` penalizes exceeding target budget.
- `mu` mildly prefers fewer steps.
- `rho` penalizes excessive remasking.

### Suggested function signature

```python
def multiplicative_budget_reward(
    exact_match: int | bool,
    nfe_used: int,
    target_budget: int,
    max_steps: int,
    remask_count: int = 0,
    beta: float = 2.0,
    mu: float = 0.5,
    rho: float = 0.1,
    eps: float = 1e-8,
) -> float:
    ...
```

### Intended use

Use this as the default reward for budget-conditioned policy training.

Wrong answers receive zero reward regardless of speed:

```text
exact_match = 0 => R = 0
```

## Config requirements

Add config fields compatible with the repository style.

Suggested YAML:

```yaml
reward:
  type: multiplicative_budget  # additive_proposal | multiplicative_budget

  additive_proposal:
    lambda_compute: 0.01
    normalize_compute: true

  multiplicative_budget:
    beta: 2.0
    mu: 0.5
    rho: 0.1
```

Or flat fields if the repo prefers that style:

```yaml
reward_type: multiplicative_budget
reward_lambda_compute: 0.01
reward_normalize_compute: true
reward_beta: 2.0
reward_mu: 0.5
reward_rho: 0.1
```

## Unit tests to create

Create tests but do not run them.

### Tests for `additive_proposal`

1. Exact formula with normalized compute:

```python
reward = additive_proposal_reward(
    task_score=1.0,
    nfe_used=32,
    max_steps=128,
    lambda_compute=0.2,
    normalize_compute=True,
)
assert reward == approx(1.0 - 0.2 * (32 / 128))
```

2. Exact formula with raw compute:

```python
reward = additive_proposal_reward(
    task_score=1.0,
    nfe_used=32,
    max_steps=128,
    lambda_compute=0.01,
    normalize_compute=False,
)
assert reward == approx(1.0 - 0.01 * 32)
```

3. Diagnostic property: fast wrong is less penalized than slow wrong.

```python
wrong_fast = additive_proposal_reward(0.0, 8, 128, 0.2, True)
wrong_slow = additive_proposal_reward(0.0, 64, 128, 0.2, True)
assert wrong_fast > wrong_slow
```

This test confirms the known risk of the proposal reward.

### Tests for `multiplicative_budget`

1. Wrong answer returns zero:

```python
assert multiplicative_budget_reward(False, 8, 16, 128) == 0.0
assert multiplicative_budget_reward(False, 64, 16, 128) == 0.0
```

2. Correct fast within budget gets higher reward than correct over-budget:

```python
within = multiplicative_budget_reward(True, 8, 16, 128)
over = multiplicative_budget_reward(True, 64, 16, 128)
assert within > over
```

3. More remasking lowers reward for correct answer:

```python
no_remask = multiplicative_budget_reward(True, 16, 16, 128, remask_count=0)
two_remasks = multiplicative_budget_reward(True, 16, 16, 128, remask_count=2)
assert no_remask > two_remasks
```

4. Larger target budget should reduce over-budget penalty for same NFE:

```python
small_budget = multiplicative_budget_reward(True, 64, 16, 128)
large_budget = multiplicative_budget_reward(True, 64, 64, 128)
assert large_budget > small_budget
```

5. No NaNs or infs for edge cases:

```python
reward = multiplicative_budget_reward(True, 0, 1, 1)
assert math.isfinite(reward)
```

## Integration tests to create, not run

- Config selects `additive_proposal`.
- Config selects `multiplicative_budget`.
- Training/eval code logs reward_type and reward parameters.
- Reward function never consumes verifier ground-truth features.
- Ground-truth exact_match is used only at reward/evaluation time.
