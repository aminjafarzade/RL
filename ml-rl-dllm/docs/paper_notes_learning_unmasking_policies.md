# Paper Notes: Learning Unmasking Policies for Diffusion Language Models

Source paper: **Learning Unmasking Policies for Diffusion Language Models**, arXiv:2512.09106v3.

These notes summarize only the implementation-relevant parts of the paper.

## Core problem

Masked diffusion language models generate by starting from a fully or partially masked sequence and iteratively unmasking tokens. The key inference-time question is:

```text
Which masked token positions should be unmasked at each denoising step?
```

Handwritten confidence heuristics can work well, but they require tuning and can degrade when block sizes get large or generation becomes fully diffusion-style rather than semi-autoregressive.

## MDP framing

The paper frames sampling as an MDP:

- **State**: prompt plus current partially masked generation.
- **Action**: token-level binary unmasking vector.
- **Transition**: frozen dLLM predicts tokens; selected masked positions are filled.
- **Reward**: final task correctness plus efficiency pressure.
- **Environment**: the frozen dLLM and decoding process.
- **Policy**: a lightweight standalone unmasking controller, not the base dLLM.

Important implementation implication:

```text
Do not update the base dLLM. Treat it as the environment.
```

## Policy architecture

The paper uses a lightweight confidence policy:

- Input includes token confidence vector, mask vector, and timestep.
- Policy is a small transformer over token positions.
- Output is one unmasking logit per position.
- Bernoulli probabilities select whether each still-masked position is unmasked.
- Already-unmasked positions are excluded from likelihood computation.
- Test-time fallback: if no position is selected, unmask the position with highest policy probability.

Implementation implication:

```text
Prefer extending the existing token-level policy rather than replacing it with only a global-ratio MLP.
```

A global-ratio MLP can be useful as a simple baseline, but the stronger method should modify the token-level policy with new features.

## GRPO training

The paper trains the lightweight unmasking policy with GRPO.

Important details:
- The dLLM sampling temperature is fixed to zero during rollouts.
- Variation among rollouts comes from unmasking decisions, not random token sampling.
- Rewards are computed at the final generation step and propagated to preceding policy decisions.
- The base dLLM remains frozen.
- The policy is trained from scratch.
- KL regularization is removed in their setup.

Implementation implication:

```text
If GRPO training code already exists, reuse it and add budget/verifier/remask features cleanly.
```

## Reward design in the paper

The paper tried an additive reward of the form:

```text
r(correctness) - alpha * compute_cost
```

It found this can produce reward hacking: fast but wrong samples may get favorable relative advantage, causing the policy to unmask too aggressively.

The paper therefore used a multiplicative correctness-gated reward, where compute only matters when the answer is correct.

Implementation implication:

```text
Implement additive proposal reward for baseline/ablation/tests, but use multiplicative/gated reward as default for final training.
```

See `docs/reward_design_and_tests.md`.

## Experiments in the paper

Main experimental setup:
- Base model: LLaDA-8B-Instruct.
- Also tested Dream-7B.
- Training mixture: GSM8K + MATH.
- Evaluation: GSM8K and MATH-500.
- Metrics: accuracy versus NFEs, and wall-clock latency.
- Generation settings:
  - semi-AR/block decoding, e.g. BL=32;
  - full diffusion, e.g. BL=L=256.

Baselines:
- random unmasking,
- high-confidence top-k,
- Fast-dLLM confidence threshold,
- learned RL policy.

Implementation implication:

```text
Start with GSM8K, then MATH/MATH-500.
Report BL=32 and BL=256 separately.
Do not mix semi-AR and full-diffusion results into one number.
```

## Key limitations to extend

The paper explicitly leaves several opportunities:

1. It trains separate policies for different speed/accuracy tradeoffs.
2. The tradeoff controlled by alpha is not smooth or fully controllable.
3. It does not implement verifier-aware remasking.
4. It mostly uses local confidence as the policy signal.
5. It suggests remasking and richer domains as future work.

Our project addresses these limitations by adding:
- explicit target/remaining budget conditioning;
- label-free verifier features;
- conservative answer-span remasking;
- confidence-vs-correctness analysis.

## Caution points for implementation

- Do not use hidden states by default; the paper found confidence-based inputs simpler and more stable.
- Do not force unmask fallback during training unless the existing code already does so safely.
- Watch for padding/EOS/EOT confidence issues, especially in full diffusion.
- Keep compute accounting fair: one frozen dLLM forward pass equals one NFE.
- If using expert steering or demonstrations, keep it optional and config-controlled.
