# Server Run Commands Template

Codex should adapt these to the actual repository entrypoints after implementation.

Do not run these locally during implementation. Run them later on the GPU server.

## 0. Environment check

```bash
cd /path/to/ml-rl-dllm
nvidia-smi
python --version
pip list | grep -E "torch|transformers|accelerate|datasets"
```

## 1. Tiny GSM8K baseline evaluation

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method high_confidence \
  --model llada-8b-instruct \
  --max_examples 16 \
  --block_length 32 \
  --max_steps 128 \
  --save_generations true \
  --log_step_traces true \
  --output_dir outputs/smoke_gsm8k_high_confidence
```

## 2. Tiny verifier-only logging

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method high_confidence \
  --model llada-8b-instruct \
  --max_examples 16 \
  --enable_verifier true \
  --verifier_type math \
  --verifier_schedule final_only \
  --enable_remasking false \
  --save_verifier_results true \
  --output_dir outputs/smoke_gsm8k_verifier_only
```

## 3. Tiny verifier-aware remasking

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method verifier_remask \
  --model llada-8b-instruct \
  --max_examples 16 \
  --enable_verifier true \
  --verifier_type math \
  --verifier_schedule after_answer_detected \
  --enable_remasking true \
  --remask_strategy verifier_answer \
  --max_remasks_per_sample 1 \
  --remask_cooldown_steps 2 \
  --verifier_threshold 0.7 \
  --output_dir outputs/smoke_gsm8k_verifier_remask
```

## 4. Full GSM8K baseline frontier

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method fast_dllm \
  --model llada-8b-instruct \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --block_length 32 \
  --max_steps 256 \
  --output_dir outputs/frontier_gsm8k_fast_dllm_bl32
```

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method fast_dllm \
  --model llada-8b-instruct \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --block_length 256 \
  --max_steps 256 \
  --output_dir outputs/frontier_gsm8k_fast_dllm_bl256
```

## 5. Best-of-N verifier reranking

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method best_of_n_verifier \
  --model llada-8b-instruct \
  --num_samples 2 4 8 \
  --match_total_nfe true \
  --enable_verifier true \
  --verifier_type math \
  --output_dir outputs/gsm8k_best_of_n_verifier
```

## 6. Budget-conditioned policy training

```bash
python -m train.train \
  --config configs/experiments/llada_budget_conditioned_gsm8k.yaml
```

## 7. Budget-conditioned policy evaluation

```bash
python -m eval.pipeline \
  --dataset gsm8k \
  --method budget_policy \
  --model llada-8b-instruct \
  --checkpoint outputs/<budget_policy_run>/checkpoints/last.pt \
  --target_budgets 8 16 32 64 128 \
  --block_length 32 \
  --max_steps 256 \
  --output_dir outputs/eval_budget_policy_gsm8k_bl32
```

## 8. MATH/MATH-500 evaluation

```bash
python -m eval.pipeline \
  --dataset math500 \
  --method budget_policy \
  --model llada-8b-instruct \
  --checkpoint outputs/<budget_policy_run>/checkpoints/last.pt \
  --target_budgets 16 32 64 128 \
  --block_length 32 \
  --max_steps 256 \
  --output_dir outputs/eval_budget_policy_math500_bl32
```

## 9. Analysis and plots

```bash
python -m eval.analyze_confidence_correctness outputs/<run_dir>
python -m eval.plot_pareto outputs/<run1> outputs/<run2> outputs/<run3>
python -m eval.collect_failure_cases outputs/<run_dir>
```

Again: these are templates. Codex should adapt them to actual repo entrypoints.
