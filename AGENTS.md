# AGENTS.md

## Project context

This repository implements or extends learned unmasking policies for diffusion language models.

The base diffusion LLM, such as LLaDA-8B-Instruct, must remain frozen. We are modifying only the lightweight decoding controller, sampler/remasking logic, verifier, reward/configs, dataset/evaluation plumbing, and logging pipeline.

This project extends the paper idea from **Learning Unmasking Policies for Diffusion Language Models** with:
1. verifier-aware math decoding,
2. conservative final-answer-span remasking,
3. budget-conditioned policy features,
4. richer analysis of confidence, verifier score, remasking fix/harm rate, and compute.

Read these files before planning or implementing:
- `docs/paper_notes_learning_unmasking_policies.md`
- `docs/project_plan_verifier_budget_remasking.md`
- `docs/reward_design_and_tests.md`
- `docs/codex_implementation_prompt.md`

Optional original PDFs may be placed under:
- `docs/papers/2512.09106v3.pdf`
- `docs/papers/proposal_verifier_budget_rl.pdf`

## Main research goal

Keep the dLLM frozen and improve the decoding controller. The controller should use confidence, entropy, mask positions, timestep, remaining/target budget, and verifier features to decide how aggressively to unmask and whether to remask suspicious final-answer spans.

The first implementation should prioritize:
- GSM8K,
- math verifier,
- answer-span remasking,
- logging,
- reward variants,
- fair compute accounting.

MATH/MATH-500 comes after GSM8K is stable. Do not add HumanEval/MBPP/code tasks in the first phase unless this repository already supports them cleanly.

## Implementation constraints

- Do not modify base LLaDA/Dream model weights.
- Reuse existing sampler, training, policy, reward, data, config, and logging abstractions.
- Do not rewrite the whole repository.
- Do not create duplicate parallel systems if equivalent modules already exist.
- Prefer minimal, clean, config-driven changes.
- Preserve existing confidence-only policy behavior when new flags are disabled.
- Count every frozen dLLM forward pass as one NFE.
- Log verifier calls separately.
- Keep remasking conservative: final numeric answer span only at first.
- Default `max_remasks_per_sample = 1`.

## Reward implementation requirement

Implement and test **both** reward variants:

### 1. Proposal additive reward

This comes from the project proposal and must be implemented for baseline/ablation/tests:

```text
R = TaskScore - lambda_compute * ComputeCost
```

For math, the simple form is:

```text
R = exact_match - lambda_compute * nfe_used
```

or the normalized form:

```text
R = exact_match - lambda_compute * (nfe_used / max_steps)
```

Use a config flag to choose normalized vs. raw compute if needed.

Important: this reward is included because it is in the proposal and should be test-covered. It is not the recommended default for final RL training because it can reward fast-but-wrong behavior.

Suggested config name:

```yaml
reward_type: additive_proposal
reward_lambda_compute: 0.01
reward_normalize_compute: true
```

### 2. Multiplicative/gated budget reward

This is the recommended default for training:

```text
R =
    exact_match
    * exp(-beta * max(0, nfe_used - target_budget) / target_budget)
    * exp(-mu * nfe_used / max_steps)
    * exp(-rho * remask_count)
```

Wrong answers must receive zero reward regardless of speed.

Suggested config:

```yaml
reward_type: multiplicative_budget
reward_beta: 2.0
reward_mu: 0.5
reward_rho: 0.1
```

## Leakage rule

Ground-truth labels may be used only for:
- training reward,
- evaluation metrics,
- offline analysis labels.

Ground-truth labels must never be used inside:
- inference-time verifier features,
- policy state,
- remasking decisions,
- heuristic keep scores,
- verifier score,
- any field consumed by the policy at inference.

The verifier may check parseability, format, arithmetic consistency, answer-span presence, and similar label-free signals. It cannot compare to the dataset answer during inference.

## Execution constraints

Do **not** run training, evaluation, pytest, smoke tests, import checks, model loading, dataset downloading, or any command that executes repository code after implementation. I will run everything later on a GPU server.

Allowed:
- read-only file inspection such as `ls`, `find`, `grep`, `sed`, `cat`, `tree`;
- editing files.

Not allowed:
- `pytest`;
- `python -m ...`;
- training/evaluation commands;
- model-loading checks;
- dataset downloads;
- import checks.

At the end of implementation, provide exact commands for me to run later on the GPU server, but do not execute them.

## Testing expectations

Create unit tests where appropriate, but do not run them.

Tests should cover:
- math verifier parsing;
- arithmetic checking;
- span finder;
- remasking strategy limits/cooldowns;
- no prompt-token remasking;
- additive reward formula;
- multiplicative reward formula;
- wrong answer zero reward for multiplicative reward;
- target-budget penalty;
- remask penalty;
- config selection between reward variants.

## Logging expectations

Per-example logs should include:
- example id,
- dataset,
- prompt,
- reference answer,
- generated text before/after remask,
- parsed answer,
- correctness before/after/final when available,
- NFE used,
- wall time,
- verifier calls,
- target budget,
- actual NFE,
- budget error,
- remask count,
- remasked token count,
- verifier score,
- parse_ok,
- format_ok,
- arithmetic_ok,
- answer span text,
- answer span token indices,
- answer span confidence,
- mean/min confidence,
- entropy stats,
- stop reason,
- seed,
- method name,
- block length,
- max steps.
