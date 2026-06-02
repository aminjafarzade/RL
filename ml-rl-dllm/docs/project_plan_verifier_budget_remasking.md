# Project Plan: Verifier-Aware Budget-Conditioned Decoding for Frozen dLLMs

## Project goal

Extend learned unmasking-policy code for diffusion LLMs by adding:

1. budget-conditioned decoding,
2. verifier-aware math signals,
3. conservative final-answer-span remasking,
4. rich logging and empirical analysis.

The base dLLM, such as LLaDA-8B-Instruct, remains frozen.

## Core hypothesis

Token confidence is local, but reasoning correctness is global. A diffusion LLM can be locally confident in a wrong reasoning token or final answer. A lightweight decoding controller may improve the quality-compute tradeoff if it sees:

- token confidence,
- entropy,
- mask positions,
- timestep,
- target/remaining budget,
- verifier features.

The project should study when these signals help, not only whether they beat every baseline.

## Primary dataset order

1. **GSM8K** first.
   - Easier numeric final-answer parsing.
   - Arithmetic verifier is meaningful.
   - Good for debugging remasking and logging.

2. **MATH/MATH-500** second.
   - Harder parsing.
   - More symbolic answers.
   - Better robustness benchmark after GSM8K works.

3. Code datasets later only if already supported.
   - HumanEval/MBPP introduce sandboxing and pass@k complexity.
   - Do not start there.

## Main implementation phases

### Phase 0: repo inspection

Identify existing:
- model loading,
- sampler/decoding loop,
- policy classes,
- GRPO trainer,
- reward functions,
- GSM8K/MATH data loaders,
- answer parsing/evaluation,
- config system,
- logging/output format,
- tests.

Do not rewrite these systems if they already exist.

### Phase 1: verifier + span finder + logging

Add a math verifier with label-free features:
- parse_ok,
- format_ok,
- arithmetic_ok,
- final_answer,
- final_answer_span_text,
- verifier_score,
- flags,
- error.

Verifier should parse:
- `#### 123`,
- `\boxed{123}`,
- `<answer>123</answer>`,
- `The answer is 123`,
- fallback last standalone number.

Arithmetic checker:
- supports simple equations with `+`, `-`, `*`, `×`, `/`, `÷`, parentheses where safe;
- never uses arbitrary `eval`;
- returns `None` when no equation is found.

Span finder:
- maps final answer text back to generated token indices;
- never remasks prompt tokens;
- never remasks padding-only spans;
- first version remasks final numeric answer span only.

Logging:
- per-example JSONL,
- verifier results,
- step traces if enabled,
- confidence/entropy stats,
- remasking actions,
- stop reason,
- budget information.

### Phase 2: verifier-aware heuristic remasking

Before verifier-aware RL, implement a conservative heuristic:

```text
If verifier_score < threshold
and final answer span is found
and remask_count < max_remasks_per_sample:
    remask final answer span
```

Defaults:
- `max_remasks_per_sample = 1`,
- `remask_cooldown_steps = 2`,
- final numeric answer only.

This tests whether verifier signal is useful before adding RL complexity.

### Phase 3: budget-conditioned policy

Add target/remaining budget features to the existing policy.

Global features:
- current step normalized,
- NFE used normalized,
- target budget normalized,
- remaining budget normalized,
- mask ratio,
- confidence summary stats,
- entropy summary stats.

Verifier features if enabled:
- parse_ok,
- format_ok,
- arithmetic_ok encoded as unknown/false/true,
- verifier_score,
- has_answer_span,
- remask_count normalized.

Token-level features if supported:
- confidence per token,
- entropy per token,
- is_masked,
- position,
- answer_span_indicator.

### Phase 4: verifier-aware RL hooks

Only after heuristic remasking shows a signal:
- add remask action/gate to the policy;
- add verifier features to policy state;
- add remask penalty;
- log fix/harm/no-change rates.

Do not overbuild verifier-aware RL before Phase 2 works.

## Reward variants

Both reward variants must be implemented and tested:

1. `additive_proposal`
   - From the proposal:
     `TaskScore - lambda * ComputeCost`.
   - Use for baseline/ablation/tests.
   - Not recommended as default final RL reward.

2. `multiplicative_budget`
   - Recommended default:
     `exact_match * exp(-beta * over_budget) * exp(-mu * nfe/max_steps) * exp(-rho * remask_count)`.
   - Wrong answers get zero reward.

See `docs/reward_design_and_tests.md`.

## Main baselines

Minimum:
- random unmasking,
- fixed-ratio unmasking,
- high-confidence top-k,
- Fast-dLLM/confidence threshold if available,
- existing learned confidence policy if available,
- verifier-aware heuristic remasking,
- best-of-N verifier reranking under matched NFE.

Ablations:
- confidence only,
- entropy only,
- budget only,
- verifier only,
- confidence + budget,
- confidence + verifier,
- full features.

## Main metrics

Quality:
- exact-match accuracy,
- parseable answer rate,
- arithmetic consistency,
- verifier score.

Compute:
- NFE,
- wall-clock time,
- verifier calls,
- remask count,
- remasked token count,
- accuracy at fixed NFE,
- target-budget compliance.

Remasking:
- fix rate,
- harm rate,
- no-change rate,
- same-wrong rate.

Analysis:
- confidence vs correctness,
- verifier score vs correctness,
- high-confidence wrong answers,
- compute by difficulty bucket,
- target budget vs actual NFE.

## Risks to handle

1. **Ground-truth leakage**
   - Never use labels as verifier features.

2. **Weak verifier**
   - Verifier may only help after final answer appears.
   - Default verifier schedule should be conservative.

3. **Remasking instability**
   - Limit remasks.
   - Do not remask same span repeatedly.

4. **Wrong reasoning chain**
   - Final-answer remasking may not fix wrong reasoning.
   - Log fix/harm/no-change.

5. **Span-finding brittleness**
   - Conservative matching only.
   - Never remask prompt/special/padding-only tokens.

6. **Budget collapse**
   - Policy may ignore target budget.
   - Log target_budget, actual_nfe, budget_error.

7. **Unfair compute comparison**
   - Count all dLLM forward passes as NFE.
   - Compare best-of-N under matched total NFE.

8. **Semi-AR vs full diffusion**
   - Keep BL=32 and BL=256 separate.

9. **GPU requirement**
   - Do not run model code locally.
   - Commands should be provided for later server execution.
