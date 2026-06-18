# Codex Implementation Prompt

You are working in an existing repository for diffusion LLM decoding / RL unmasking policies.

## Critical execution constraint

Do not run training, evaluation, pytest, smoke tests, Python import checks, model loading, dataset downloading, or any command that executes repository code after implementation.

I will run the updated code later on a GPU server.

You may inspect files using safe read-only shell commands such as:

```bash
ls
find
grep
sed
cat
tree
```

You may edit files. Do not run GPU code. Do not run CPU smoke tests either. At the end, provide exact commands that I should run later on the server, but do not execute them.

## Read first

Before editing code, read:

- `AGENTS.md`
- `docs/paper_notes_learning_unmasking_policies.md`
- `docs/project_plan_verifier_budget_remasking.md`
- `docs/reward_design_and_tests.md`

The paper notes describe the existing method we are extending. The project plan describes the new method. Do not simply reimplement the paper. Extend the existing repository minimally to support verifier-aware remasking, budget-conditioned policy features, both reward variants, and richer evaluation logging.

## Project title

Verifier-Aware Budget-Conditioned Decoding for Frozen Diffusion LLMs

## Project context

This project extends a diffusion language model decoding / RL unmasking-policy repository, likely based on **Learning Unmasking Policies for Diffusion Language Models**.

The base diffusion LLM, such as LLaDA-8B-Instruct, must remain frozen. The goal is to modify only:

- lightweight decoding controller,
- sampler,
- verifier,
- remasking strategy,
- reward functions,
- configs,
- evaluation/logging pipeline.

## Research idea

Masked diffusion LLMs generate by iteratively unmasking tokens. Existing confidence-based decoding can be locally confident but globally wrong on reasoning tasks. We want to study whether explicit budget conditioning and task-level verifier signals can improve decoding under fixed compute.

## Main hypothesis

A frozen diffusion LLM decoder can be improved by a lightweight controller that observes confidence, entropy, mask positions, timestep, remaining/target budget, and verifier features. The controller should decide how aggressively to unmask and whether to remask suspicious final-answer spans.

The project should prioritize careful empirical analysis over claiming state of the art.

## Primary model

Use LLaDA-8B-Instruct as the frozen base dLLM/backend if this repo supports it. Do not modify base model weights.

## Primary datasets

Start with GSM8K for development and main debugging.

Use MATH/MATH-500 as harder evaluation after GSM8K works.

Do not add HumanEval, MBPP, or code-generation datasets in the first implementation phase unless the repository already supports them cleanly.

## Dataset handling

First inspect whether the repository already has GSM8K, MATH, or MATH-500 loaders, reward functions, answer parsers, or evaluation utilities.

If they exist, reuse them.

If they do not exist, add loaders following the repository’s existing structure, likely under `data/`, `datasets/`, `common/data/`, or similar.

Dataset paths must be configurable through YAML/config files or existing config mechanisms. Do not hardcode local dataset paths.

## Implementation mode

First inspect the repository structure and produce a short implementation plan internally. Then implement the minimal clean changes.

Do not rewrite the whole codebase.

Preserve existing abstractions.

Do not create parallel duplicate systems if equivalent modules already exist.

## Implementation phases

### Phase 1: Math verifier, span finder, and logging

Add a math verifier abstraction.

Suggested files, but adapt to actual repo:

- `common/verifiers/base.py`
- `common/verifiers/math_verifier.py`
- `common/remasking/span_finder.py`
- `common/remasking/strategies.py`
- `tests/test_math_verifier.py`
- `tests/test_span_finder.py`
- `tests/test_remasking_strategy.py`

Do not run the tests. Only create them.

Base verifier API:

```python
@dataclass
class VerifierResult:
    parse_ok: bool
    format_ok: bool
    arithmetic_ok: Optional[bool]
    final_answer: Optional[str]
    final_answer_span_text: Optional[str]
    verifier_score: float
    flags: dict[str, Any]
    error: Optional[str]

class BaseVerifier:
    def verify(self, text: str, prompt: Optional[str] = None) -> VerifierResult:
        ...
```

Math verifier requirements:

- Extract final answers from:
  - `#### 123`
  - `\boxed{123}`
  - `<answer>123</answer>`
  - `The answer is 123`
  - last standalone number as fallback
- Normalize integers, decimals, fractions, commas, currency symbols, and simple LaTeX numeric forms.
- Check `format_ok` when a recognized final-answer format exists.
- Extract simple arithmetic equations such as:
  - `48 + 24 = 72`
  - `12 * 4 = 48`
  - `6 × 4 = 24`
  - `300 - 12 = 288`
- Support `+`, `-`, `*`, `×`, `/`, `÷`, and parentheses where safe.
- Use a safe arithmetic evaluator. Never use `eval` on arbitrary text.
- `arithmetic_ok` should be:
  - `True` if parsed equations are valid,
  - `False` if at least one parsed equation is invalid,
  - `None` if no equation is found.
- `verifier_score` should be label-free and based only on parse_ok, format_ok, arithmetic_ok, and related non-ground-truth signals.
- Ground-truth answers must never be used inside verifier_score or verifier features.
- Malformed text must not crash decoding.

Critical leakage rule:

Ground-truth labels may only be used for training rewards and final evaluation metrics.

Ground-truth labels must never be included in:
- inference-time verifier features,
- policy state,
- remasking decisions,
- heuristic keep scores,
- logs that are consumed by the policy at inference.

Span finder requirements:

- Given tokenizer, decoded text, generated token IDs, and VerifierResult, return token indices for the final answer span.
- Start conservatively:
  - exact substring matching using `final_answer_span_text`,
  - map character offsets to token offsets if tokenizer supports offset mappings,
  - otherwise use decode-prefix matching.
- Fallback:
  - remask the smallest suffix around the final numeric answer.
- Never remask prompt tokens.
- Never remask special tokens unless explicitly configured.
- Never remask padding-only spans.
- Do not remask reasoning tokens in the first version.
- First version should target only final numeric answer spans.

Remasking strategies:

Implement:
- `NoRemask`
- `LowConfidenceRemask`, if easy and consistent with repo style
- `VerifierAnswerRemask`

`VerifierAnswerRemask` behavior:

- If `verifier_score < threshold` and final answer span is found, remask that span.
- Optional condition: target cases where answer-span confidence is high but verifier is inconsistent.
- Configurable:
  - `max_remasks_per_sample`
  - `remask_cooldown_steps`
  - `min_steps_before_remask`
  - `remask_span_mode`
  - `verifier_threshold`
- Default `max_remasks_per_sample = 1`.
- Default `remask_cooldown_steps = 2`.
- Prevent remasking the same span repeatedly.
- Stop when budget is exhausted.

### Phase 2: Sampler integration and logging

Find the existing sampler/decoding loop. Extend it rather than replacing it.

Track:
- current token IDs,
- mask positions,
- model confidences,
- entropy if available,
- unmasked token indices,
- remasked token indices,
- verifier result when verifier is called,
- NFE count,
- wall-clock time,
- stop reason,
- EOS/EOT/padding positions if available.

Add config-driven verifier scheduling:
- `none`
- `final_only`
- `after_answer_detected`
- `every_k_steps`
- `final_block_only`

Default:
- `after_answer_detected` or `final_only`, whichever is easier and safer.
- Do not call verifier at every step by default.

Sampler logic:
1. Run frozen dLLM forward pass.
2. Increment NFE count.
3. Compute predictions, confidence, entropy.
4. Choose unmasking action using existing baseline/policy.
5. Apply unmasking.
6. If verifier is enabled and schedule says to check:
   - decode current generation,
   - run math verifier,
   - find answer span,
   - optionally remask answer span.
7. Stop if all tokens are unmasked, EOS/EOT stopping condition is satisfied, max steps reached, or target budget exhausted.

Compute accounting:
- Every frozen dLLM forward pass counts as one NFE.
- Remasking itself does not count as NFE, but subsequent model calls do.
- Log verifier_calls separately.
- Log wall-clock time, but do not rely on it as the main metric.

EOS/EOT/padding handling:
- Log EOS/EOT position if available.
- Log compute spent after EOS.
- Do not allow verifier remasking to target padding-only spans.
- Preserve existing EOS/EOT handling unless clearly broken.

Output logging:

For each evaluation run, save:
- `metrics.json`
- `per_example.jsonl`
- `generations.jsonl`
- `verifier_results.jsonl`
- `per_step_traces.jsonl` if enabled
- config copy
- checkpoint metadata if applicable

`per_example.jsonl` fields:
- `example_id`
- `dataset`
- `prompt`
- `reference_answer`
- `generated_text_before_remask`
- `generated_text_after_remask`
- `generated_text_final`
- `parsed_answer`
- `correct_before_remask`, if available
- `correct_after_remask`, if available
- `correct_final`
- `nfe_used`
- `wall_time`
- `target_budget`
- `budget_error`
- `remask_count`
- `remasked_token_count`
- `verifier_calls`
- `final_verifier_score`
- `parse_ok`
- `format_ok`
- `arithmetic_ok`
- `answer_span_text`
- `answer_span_token_indices`
- `answer_span_mean_confidence`
- `answer_span_min_confidence`
- `mean_confidence`
- `min_confidence`
- `entropy_stats`
- `stop_reason`
- `seed`
- `method_name`
- `block_length`
- `max_steps`

### Phase 3: Budget-conditioned features and reward

Add budget-conditioned policy features, but preserve existing confidence-only behavior when disabled.

Global features:
- current_step_normalized,
- nfe_used_normalized,
- target_budget_normalized,
- remaining_budget_normalized,
- mask_ratio,
- mean_confidence,
- min_confidence,
- max_confidence,
- mean_entropy,
- min_entropy,
- max_entropy.

Verifier features, if enabled:
- parse_ok as float,
- format_ok as float,
- arithmetic_ok encoded as unknown/false/true,
- verifier_score,
- has_answer_span,
- remask_count_normalized.

Token-level features, if existing policy supports token-level input:
- confidence per position,
- entropy per position if available,
- is_masked,
- normalized position or existing positional encoding,
- answer_span_indicator,
- optionally last_unmask_step / last_remask_step if histories already exist.

Architecture:
- Prefer extending the existing token-level policy from the paper/repo.
- If existing policy is a transformer over token positions, broadcast global budget/verifier features through concatenation, conditioning MLP, or existing AdaLN-style conditioning.
- A global-ratio MLP policy can be added as a simple baseline, but should not replace the stronger token-level policy unless repo constraints require it.

### Reward implementation

Implement and test both reward variants.

#### `additive_proposal`

Formula:

```text
R = TaskScore - lambda_compute * ComputeCost
```

For math:

```text
R = exact_match - lambda_compute * nfe_used
```

or normalized:

```text
R = exact_match - lambda_compute * (nfe_used / max_steps)
```

Use this for:
- proposal fidelity,
- baseline/ablation,
- diagnostic tests.

Do not make it the default final RL reward.

#### `multiplicative_budget`

Recommended default:

```text
R =
    exact_match
    * exp(-beta * max(0, nfe_used - target_budget) / target_budget)
    * exp(-mu * nfe_used / max_steps)
    * exp(-rho * remask_count)
```

Where:
- `exact_match` is 1 if final parsed answer matches dataset answer, otherwise 0.
- `nfe_used` is dLLM forward passes.
- `target_budget` is sampled during budget-conditioned training.
- `max_steps` is max denoising horizon.
- `remask_count` is number of remasking actions.
- `beta` controls penalty for exceeding target budget.
- `mu` controls mild general preference for fewer steps.
- `rho` penalizes excessive remasking.

Default values:
- `beta = 2.0`
- `mu = 0.5`
- `rho = 0.1`

Important:
- Wrong answers should receive zero reward regardless of speed for multiplicative reward.
- Verifier features may be used as state features.
- `verifier_score` should not be the main training reward in the first implementation.
- Ground-truth exact match may be used for training reward and evaluation only.

Budget-conditioned training:
- Sample target_budget from a config list, e.g. `[8, 16, 32, 64, 128]`.
- Add target_budget and remaining_budget to policy features.
- Log:
  - target_budget,
  - actual_nfe,
  - budget_error,
  - exact_match,
  - reward,
  - reward_type,
  - remask_count.
- The budget-conditioned policy is only useful if actual_nfe responds to target_budget. Add evaluation logging to diagnose this.

### Phase 4: Evaluation and analysis scripts

Add or extend evaluation scripts to support:

Main methods:
- random unmasking,
- fixed-ratio unmasking,
- high-confidence top-k,
- Fast-dLLM / confidence threshold if available,
- existing learned confidence policy if available,
- verifier-aware heuristic remasking,
- best-of-N verifier reranking under matched total NFE,
- budget-conditioned policy,
- verifier-aware RL policy hooks only if easy.

Main comparisons:
1. Baseline frontier:
   - random,
   - high-confidence,
   - Fast-dLLM/confidence threshold,
   - existing learned policy.

2. Verifier-aware heuristic:
   - confidence-only decoding,
   - verifier-aware answer-span remasking,
   - low-confidence remasking, if easy,
   - best-of-N verifier reranking with matched NFE.

3. Budget-conditioned policy:
   - evaluate one policy at target budgets `[8, 16, 32, 64, 128]`,
   - compare to fixed schedules and confidence thresholds.

4. Ablations:
   - confidence only,
   - entropy only,
   - budget only,
   - verifier only,
   - confidence + budget,
   - confidence + verifier,
   - full features.

5. Failure analysis:
   - confidence-only wrong and high-confidence,
   - verifier remasking fixes,
   - verifier remasking harms,
   - verifier false positive,
   - verifier false negative,
   - best-of-N beats online remasking,
   - policy stops too early,
   - policy remasks too aggressively.

Metrics:
- exact-match accuracy,
- parseable answer rate,
- arithmetic consistency rate,
- verifier score,
- average NFE,
- median NFE,
- wall-clock latency,
- verifier_calls,
- remask_count,
- remasked_token_count,
- accuracy at fixed NFE,
- target budget compliance,
- fix rate,
- harm rate,
- no-change rate,
- confidence AUROC/AUPRC for correctness if easy,
- verifier AUROC/AUPRC for correctness if easy.

Plots to generate later on the server:
- accuracy vs NFE,
- accuracy vs wall-clock,
- target budget vs actual NFE,
- confidence reliability curve,
- verifier reliability curve,
- fix/harm/no-change stacked bar,
- compute by difficulty bucket,
- remask rate by verifier score,
- token unmask/remask trajectory heatmap.

Do not generate plots now. Only implement scripts or placeholders needed to generate them later.

## Config additions

Add config fields through the repository’s existing config system:

```yaml
method: high_confidence  # random | fixed_ratio | high_confidence | fast_dllm | learned_policy | verifier_remask | budget_policy | verifier_policy

dataset: gsm8k  # gsm8k | math | math500

enable_budget_conditioning: false
target_budget: 32
target_budgets: [8, 16, 32, 64, 128]
train_budget_sampling: [8, 16, 32, 64, 128]

enable_verifier: false
verifier_type: math
verifier_schedule: final_only
verifier_every_k_steps: 8
verifier_threshold: 0.7

enable_remasking: false
remask_strategy: none
max_remasks_per_sample: 1
remask_cooldown_steps: 2
min_steps_before_remask: 0
remask_answer_span_mode: numeric_only
remask_penalty: 0.1

reward_type: multiplicative_budget
reward_lambda_compute: 0.01
reward_normalize_compute: true
reward_beta: 2.0
reward_mu: 0.5
reward_rho: 0.1

log_step_traces: false
save_generations: true
save_verifier_results: true
seed: 0
block_length: 32
max_steps: 256
count_verifier_time: true
```

Adapt to repo style.

## Testing files

Create unit tests, but do not run them.

Tests to add:
- math verifier parses `####`, boxed answer, answer tags, and fallback final number;
- math verifier normalizes commas, decimals, fractions, currency;
- arithmetic checker identifies valid and invalid equations;
- arithmetic checker rejects malicious or invalid expressions safely;
- span finder maps simple final answers to token spans;
- remasking strategy respects `max_remasks_per_sample`;
- remasking strategy respects cooldown;
- sampler does not remask prompt tokens;
- budget feature builder returns expected fields/shapes;
- additive proposal reward formula;
- additive proposal reward diagnostic behavior for wrong_fast vs wrong_slow;
- multiplicative reward returns zero for wrong answers regardless of NFE;
- multiplicative reward decreases for correct answers that exceed target budget;
- multiplicative reward decreases with remask_count;
- config selects each reward variant.

Do not run pytest.

## Server-run command templates

At the end of implementation, provide commands I can run later on the GPU server. Adapt them to actual repo entrypoints. Include commands for:

1. install/environment check,
2. tiny GSM8K baseline evaluation,
3. verifier-only logging on tiny GSM8K subset,
4. verifier-aware heuristic remasking on tiny GSM8K subset,
5. full GSM8K baseline frontier,
6. MATH/MATH-500 evaluation,
7. budget-conditioned policy training,
8. budget-conditioned policy evaluation,
9. analysis/plot generation.

Do not execute any commands.

## Final response required from Codex after implementation

- Summary of files modified/created.
- Explanation of key design choices.
- Any assumptions made about repository structure.
- Any TODOs or uncertain integration points.
- Exact server commands to run later.
- Reminder that tests/training/evaluation were not run because GPU/server execution is required.
