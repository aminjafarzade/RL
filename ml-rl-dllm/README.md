# Budget-Aware Neural Continuation Policies for Frozen Diffusion LLMs

This repository studies **adaptive inference for diffusion language models**. The project is inspired by the paper **“Learning Unmasking Policies for Diffusion Language Models”**, which trains a lightweight RL policy on top of a frozen diffusion LLM to control token unmasking decisions.

Our final direction is different but related: instead of learning a token-level unmasking policy, we train a small **neural contextual-bandit controller** that decides whether a frozen LLaDA generation should receive extra answer-span continuation compute.

In short:

```text
Frozen LLaDA base answer
        ↓
Extract base-generation features
        ↓
Neural contextual bandit
        ↓
Decision: stop or continue
```

The goal is not to fine-tune LLaDA or directly improve its reasoning ability. The goal is to improve the **compute-quality tradeoff** during inference: spend continuation compute only when it is likely to help.

---

## Motivation

The reference paper formulates masked diffusion sampling as an RL problem. The frozen diffusion LLM is treated as the environment, and a lightweight policy learns which masked tokens to unmask at each diffusion step. This is a token-level control problem.

We initially explored a paper-style GRPO/token-level unmasking direction, but in our stricter GSM8K setting the reward signal was too sparse. Many rollout groups had all-zero or near-identical rewards, producing weak or zero group-relative advantage signals. This made token-level GRPO unstable for our deadline.

However, continuation analysis showed a useful signal:

- Extra answer-span continuation can fix wrong base answers.
- But blindly continuing can also harm already-correct answers.

So we reformulated the problem as a simpler one-step adaptive-compute decision:

```text
Should we stop after the base answer, or continue/refine the answer span?
```

This gives a stable offline contextual-bandit problem.

---

## Method

### State

The state is a vector of non-leaky features extracted from the base generation, such as confidence, verifier, formatting, budget, repetition, and answer-span signals.

Outcome/leaky fields are **not** used as input features. In particular, the policy must not see:

```text
base_correct
continued_correct
continuation_gain
label_gain_positive
label_correct_gain
label_harm
reference_answer
prompt
generated_text
base_generated_text
continued_generated_text
answer_quality
task_score
reward
correctness
ground_truth
answer
continued_extra_steps
actual_extra_steps
```

These fields may be used only for reward construction and evaluation.

### Action

The action space is binary:

```text
0 = stop and keep the base answer
1 = run K=4 answer-span continuation
```

### Reward

For fixed-lambda training, each example has two logged outcomes: stop and continue.

```text
r_stop = base_correct

r_continue = continued_correct
             - cost_lambda * actual_extra_steps
             - harm_penalty * I[base_correct and not continued_correct]
```

The oracle continuation decision is:

```text
oracle_continue = r_continue > r_stop
```

### Neural Bandit Architecture

The policy is a small MLP Bernoulli controller:

```text
input features
  → Linear(input_dim, 64)
  → ReLU
  → Dropout(0.1)
  → Linear(64, 64)
  → ReLU
  → Dropout(0.1)
  → Linear(64, 1)
  → sigmoid
  → p_continue
```

Main hyperparameters:

```text
hidden_dim = 64
num_layers = 2
dropout = 0.1
learning_rate = 1e-3
epochs = 300
patience = 40
entropy_coef = 0.01
aux_bce_weight = 0.25
bootstrap_samples = 100
```

### Training Objective

The policy is trained offline using a full-information contextual-bandit objective:

```text
E[R] = p_continue * r_continue + (1 - p_continue) * r_stop
```

The loss is:

```text
loss = -mean(E[R])
       - entropy_coef * entropy(p_continue)
       + aux_bce_weight * BCE(logit, oracle_continue)
```

This produces an actual neural policy checkpoint, for example:

```text
neural_bandit_policy.pt
```

---

## Budget-Conditioned Extension

The reference paper notes that controlling the speed/accuracy tradeoff may require separate policies for different compute settings. As an extension, we train one policy conditioned on a requested compute cost:

```text
πθ(continue | base_features, requested_cost_lambda)
```

During training, each row is duplicated for multiple cost values:

```text
[0.0, 0.0025, 0.005, 0.01, 0.02]
```

The feature `requested_cost_lambda` is added to the input, and the reward is recomputed for each value. This gives one checkpoint that can operate at several accuracy/compute points at test time.

---

## Main Results

Final evaluation uses 512 held-out examples.

### Fixed-Lambda Neural Bandit

Always-continuation is not Pareto optimal: the neural bandit reaches equal or better accuracy with less continuation compute.

| Method | Accuracy | Avg Extra Steps | Compute Saved vs Always | Fixes | Harms | Net |
|---|---:|---:|---:|---:|---:|---:|
| never_continue | 0.1172 | 0.0000 | 100.0% | 0 | 0 | 0 |
| always_continue | 0.1406 | 2.0625 | 0.0% | 25 | 13 | 12 |
| combined_heuristic | 0.1387 | 2.0391 | 1.1% | 24 | 13 | 11 |
| neural λ=0 | **0.1465** | 1.3965 | **32.3%** | 23 | 8 | **15** |
| neural λ=0.005 | **0.1445** | 1.0000 | **51.5%** | 20 | 6 | 14 |
| neural λ=0.01 | 0.1406 | 0.7441 | **63.9%** | 19 | 7 | 12 |
| neural λ=0.02 | 0.1387 | 0.4922 | **76.1%** | 14 | 3 | 11 |
| utility_oracle | 0.1660 | 0.1289 | 93.8% | 25 | 0 | 25 |

Key takeaways:

- `neural λ=0` improves accuracy over always-continuation from **0.1406 → 0.1465** while saving **32.3%** continuation compute.
- `neural λ=0.005` achieves **0.1445** accuracy with **51.5%** less continuation compute.
- `neural λ=0.01` matches always-continuation accuracy while saving **63.9%** continuation compute.
- Always-continuation fixes more raw examples, but it also causes more harms. The learned policy trades a few fixes for fewer harmful continuations.

For example:

```text
always_continue: fixes = 25, harms = 13, net = 12
neural λ=0:      fixes = 23, harms = 8,  net = 15
```

So the policy misses 2 possible fixes but avoids 5 harmful continuations.

---

### Budget-Conditioned Neural Bandit

One checkpoint is trained with `requested_cost_lambda` as an input and evaluated at several operating points.

With per-lambda validation thresholds:

| λ | Accuracy | Utility | Trigger Rate | Avg Extra Steps | Saved Steps vs Always | Approx Saving | Fixes | Harms | Net |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.1484** | 0.1484 | 0.6055 | 1.1543 | 0.9082 | 44.0% | 21 | 5 | 16 |
| 0.0025 | **0.1484** | 0.1455 | 0.6074 | 1.1562 | 0.9062 | 43.9% | 21 | 5 | 16 |
| 0.005 | 0.1465 | 0.1406 | 0.6191 | 1.1680 | 0.8945 | 43.4% | 21 | 6 | 15 |
| 0.01 | 0.1465 | 0.1366 | 0.5176 | 0.9883 | 1.0742 | 52.1% | 20 | 5 | 15 |
| 0.02 | 0.1445 | 0.1264 | 0.4668 | 0.9082 | 1.1543 | 56.0% | 20 | 6 | 14 |

Key takeaway:

A single budget-conditioned checkpoint improves over always-continuation at every tested operating point. At λ=0, it reaches **0.1484** accuracy with **44.0%** less compute. At λ=0.02, it still reaches **0.1445** accuracy with **56.0%** less compute.

---

### Fixed-Threshold Budget-Conditioned Ablation

To verify that `requested_cost_lambda` itself controls behavior, we evaluated the budget-conditioned model with a fixed threshold of 0.5 for every λ.

| λ | Accuracy | Avg Extra Steps | Trigger Rate | Saved Steps vs Always | Approx Saving | Fixes | Harms | Net |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1445 | 1.0215 | 0.5215 | 1.0410 | 50.5% | 20 | 6 | 14 |
| 0.0025 | 0.1445 | 0.9336 | 0.4883 | 1.1289 | 54.7% | 20 | 6 | 14 |
| 0.005 | 0.1445 | 0.8691 | 0.4492 | 1.1934 | 57.9% | 20 | 6 | 14 |
| 0.01 | 0.1465 | 0.7656 | 0.3984 | 1.2969 | 62.9% | 20 | 5 | 15 |
| 0.02 | 0.1465 | 0.6035 | 0.3164 | 1.4590 | 70.7% | 20 | 5 | 15 |

This ablation shows monotonic compute control:

```text
λ increases:         0 → 0.0025 → 0.005 → 0.01 → 0.02
trigger rate:        0.5215 → 0.4883 → 0.4492 → 0.3984 → 0.3164
avg extra steps:     1.0215 → 0.9336 → 0.8691 → 0.7656 → 0.6035
```

Accuracy remains above the always-continuation baseline.

---

## Repository Structure

Important files and directories:

```text
configs/experiment_configs/
  llada_8b_instruct_verifier_budget_gsm8k_continuation_k4_fresh.yaml

scripts/
  run_continuation_analysis.py
  export_continuation_gate_dataset.py
  train_contextual_bandit_continuation_policy.py
  train_neural_continuation_bandit.py
  train_budget_conditioned_neural_bandit.py
  check_training_consistency.py

tests/
  test_contextual_bandit_continuation_policy.py
  test_continuation_analysis_scripts.py
  test_neural_continuation_bandit.py
  test_budget_conditioned_neural_bandit.py

outputs/
  llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/
    continuation_gate_dataset.jsonl
    neural_bandit_*/
    budget_conditioned_neural_bandit/
    budget_conditioned_neural_bandit_fixed05/
```

---

## Environment Setup

Create and activate an environment:

```bash
conda create -n rl_dllm python=3.10 -y
conda activate rl_dllm
```

Install dependencies. If the repository has a requirements file, use it:

```bash
pip install -r requirements.txt
```

If there is no requirements file, install the core packages manually:

```bash
pip install torch transformers accelerate datasets numpy pandas scikit-learn scipy tqdm pyyaml joblib pytest
```

For LLaDA generation, install a CUDA-enabled PyTorch build compatible with your GPU. For the bandit training scripts, CPU is usually enough because the model is small.

Set the Python path from the repository root:

```bash
export PYTHONPATH=.
```

If the model is downloaded from Hugging Face, authenticate if necessary:

```bash
huggingface-cli login
```

Base model used by the configs:

```text
GSAI-ML/LLaDA-8B-Instruct
```

---

## Running Tests

Run the main CPU tests:

```bash
PYTHONPATH=. pytest \
  tests/test_neural_continuation_bandit.py \
  tests/test_budget_conditioned_neural_bandit.py \
  tests/test_contextual_bandit_continuation_policy.py \
  tests/test_continuation_analysis_scripts.py \
  -q
```

---

## Reproducing the Experiments

There are two levels of reproduction:

1. **Fast reproduction**: use the already exported `continuation_gate_dataset.jsonl` and train/evaluate the bandit policies.
2. **Full reproduction**: rerun frozen LLaDA continuation analysis to regenerate the dataset, then train the policies.

### 1. Full Continuation Analysis

This is the expensive step because it runs frozen LLaDA.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/run_continuation_analysis.py \
  --config configs/experiment_configs/llada_8b_instruct_verifier_budget_gsm8k_continuation_k4_fresh.yaml \
  --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis \
  --split both \
  --skip_failed_batches
```

Then export the gate dataset:

```bash
PYTHONPATH=. python scripts/export_continuation_gate_dataset.py \
  --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis
```

Expected dataset:

```text
outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/continuation_gate_dataset.jsonl
```

### 2. Train Fixed-Lambda Neural Bandits

Accuracy-focused model:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_neural_continuation_bandit.py \
  --dataset outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/continuation_gate_dataset.jsonl \
  --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/neural_bandit_lambda0 \
  --cost_lambda 0.0 \
  --harm_penalty 0.0 \
  --epochs 300 \
  --patience 40 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --entropy_coef 0.01 \
  --aux_bce_weight 0.25 \
  --bootstrap_samples 100 \
  --device cpu
```

Compute-aware grid:

```bash
for L in 0.005 0.01 0.02; do
  PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_neural_continuation_bandit.py \
    --dataset outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/continuation_gate_dataset.jsonl \
    --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/neural_bandit_lambda_${L} \
    --cost_lambda $L \
    --harm_penalty 0.0 \
    --epochs 300 \
    --patience 40 \
    --lr 1e-3 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1 \
    --entropy_coef 0.01 \
    --aux_bce_weight 0.25 \
    --bootstrap_samples 100 \
    --device cpu
 done
```

Print results:

```bash
for f in outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/neural_bandit*/neural_bandit_result_pretty.md; do
  echo "================ $f ================"
  cat "$f"
done
```

### 3. Train Budget-Conditioned Neural Bandit

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_budget_conditioned_neural_bandit.py \
  --dataset outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/continuation_gate_dataset.jsonl \
  --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/budget_conditioned_neural_bandit \
  --train_cost_lambdas "0.0,0.0025,0.005,0.01,0.02" \
  --eval_cost_lambdas "0.0,0.0025,0.005,0.01,0.02" \
  --harm_penalty 0.0 \
  --epochs 300 \
  --patience 40 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --entropy_coef 0.01 \
  --aux_bce_weight 0.25 \
  --bootstrap_samples 100 \
  --device cpu
```

Print result:

```bash
cat outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/budget_conditioned_neural_bandit/budget_conditioned_result_pretty.md
```

### 4. Fixed-Threshold Budget-Conditioned Ablation

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_budget_conditioned_neural_bandit.py \
  --dataset outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/continuation_gate_dataset.jsonl \
  --output_dir outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/budget_conditioned_neural_bandit_fixed05 \
  --train_cost_lambdas "0.0,0.0025,0.005,0.01,0.02" \
  --eval_cost_lambdas "0.0,0.0025,0.005,0.01,0.02" \
  --harm_penalty 0.0 \
  --epochs 300 \
  --patience 40 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.1 \
  --entropy_coef 0.01 \
  --aux_bce_weight 0.25 \
  --threshold_mode fixed_0_5 \
  --bootstrap_samples 100 \
  --device cpu
```

Print result:

```bash
cat outputs/llada_verifier_budget_gsm8k_continuation_k4_fresh_analysis/budget_conditioned_neural_bandit_fixed05/budget_conditioned_result_pretty.md
```

---

## Metrics

- **Accuracy**: fraction of final answers whose extracted answer matches the GSM8K ground truth.
- **Utility**: reward after including compute cost.
- **Trigger rate**: fraction of examples where the policy chooses continuation.
- **Avg extra steps**: average actual continuation steps used per example.
- **Compute saved vs always**: reduction in average extra steps compared with always-continuation.
- **Fixes**: base answer wrong, continued answer correct, and policy continued.
- **Harms**: base answer correct, continued answer wrong, and policy continued.
- **Net**: fixes minus harms.
- **Utility oracle**: an upper bound that chooses continue iff `r_continue > r_stop`; not deployable because it uses both outcomes.

---

## Notes and Troubleshooting

### CPU vs GPU

The neural bandit is tiny and usually trains in seconds on CPU. GPU is not required for bandit training.

Use GPU mainly for LLaDA generation/continuation analysis.

If you see a CUDA warning such as:

```text
NVIDIA Graphics Device with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

then your PyTorch build does not support that GPU architecture. For bandit training, switch to:

```bash
--device cpu
```

For LLaDA generation, install a compatible PyTorch/CUDA build or use another GPU node.

### Shell continuation

When using multiline bash commands, the backslash `\` must be the last character on the line. If the terminal shows `>`, the shell is waiting for the rest of the command.

---

## Final Project Takeaway

The main finding is:

> Continuation is useful but dangerous. Blindly applying continuation improves some answers but also breaks others. A neural contextual-bandit controller can learn when continuation is worth the compute, improving the accuracy/compute tradeoff over always-continuation.

The strongest result is the budget-conditioned extension: one checkpoint controls multiple compute settings and outperforms always-continuation across all tested operating points.

---

## Citation

Reference paper:

```bibtex
@article{jazbec2026learning,
  title={Learning Unmasking Policies for Diffusion Language Models},
  author={Jazbec, Metod and Olausson, Theo X. and Bethune, Louis and Ablin, Pierre and Kirchhof, Michael and Monteiro, Joao and Turrisi, Victor and Ramapuram, Jason and Cuturi, Marco},
  journal={arXiv preprint arXiv:2512.09106},
  year={2026}
}
```
