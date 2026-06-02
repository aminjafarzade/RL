# Learning Unmasking Policies for Diffusion Language Models

This software project accompanies the research paper, [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106).

To summarize the work very briefly:
Diffusion LLMs generate text by iteratively unmasking tokens. Most prior work use heuristics to decide which tokens to unmask at each step.
This work instead learns a lightweight policy (via GRPO) that makes these decisions autonomously, formalizing the unmasking problem as Markov Decision Process where the frozen dLLM serves as the environment.
The policy, which we implement as a DiT-style, single-block transformer, observes token confidences and outputs per-position unmasking probabilities.
If you are interested in learning more, please follow the link to the paper above.

If you find this work useful, please cite the paper:

```bibtex
@misc{jazbec2025learningunmaskingpoliciesdiffusion,
      title={Learning Unmasking Policies for Diffusion Language Models},
      author={Metod Jazbec and Theo X. Olausson and Louis Béthune and Pierre Ablin and Michael Kirchhof and Jo\~ao Monteiro and Victor Turrisi and Jason Ramapuram and Marco Cuturi},
      year={2025},
      eprint={2512.09106},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.09106},
}
```

## Getting Started

The code requires Python 3.12. To install:

```bash
uv pip install -e .   # or just `pip install -e.` if uv is not available
```

You will also need to set your Hugging Face token for model access:

```bash
export HF_TOKEN=<your-token>
```

## Quick Start

To train a policy on LLaDA-8B-Instruct with the GSM8k+MATH mixture:

```bash
python -m train.train --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
```

To evaluate a trained policy:

```bash
python -m eval.pipeline ./outputs/my_experiment \
    configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --checkpoints last \
    --datasets gsm8k \
    --temperatures 1.0 \
    --sampling_mode bernoulli-argmax
```

Here, `./outputs/my_experiment` should be a directory containing `checkpoint-*` subdirectories (as created by training).

See more details below.

## Supported Models and Datasets

**Models**:

- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [Dream-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)

**Training datasets**: GSM8k, MATH, KodCode (or mixtures thereof).

**Evaluation datasets**: GSM8k, MATH-500, HumanEval, MBPP.

## Training

As stated in the Quick Start section, the main training interface is `train.train`, which takes all of its parameters in the input yaml config file.

Key config parameters are:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to the underlying dLLM (e.g., `GSAI-ML/LLaDA-8B-Instruct`) |
| `block_length` | `int` | Token block size for semi-AR generation; we use 32 or 256 in the paper. Setting `block_length` equal to the generation length disables semi-AR |
| `policy_type` | `str` | We use `dit_confidence` for all main experiments; `dit_hidden` is only used for ablations |
| `reward_functions` | `list[str]` | We use `mixed_correctness_mult_reward_func` for all main experiments; additive variant is for ablations only |
| `reward_weights` | `list[float]` | Weights for each reward function (in order); set to 0 to log a reward to wandb without affecting the loss |
| `alpha_compute_reward` | `float` | Corresponds to alpha in the paper; higher values force the policy to favor efficiency over accuracy |
| `sampling_mode` | `str` | We use `bernoulli` or `dpls` for training, and `bernoulli-argmax` for eval if trained with `bernoulli`. See Appendix C in the paper for details on DPLS. |
| `temperature` | `float` | Policy temperature; scales logits before sampling. We use 0.5 for `block_length=32` and 1.0 for `block_length=256` |
| `policy_smart_init` | `float` | Sets the bias of the output layer of the policy. Lower values means a slower (and thus more likely to yield correct answers) policy at the start of training. We use -2.0 for all experiments. |

See `configs/experiment_configs/` for more full-fledged examples.

For multi-GPU training, you can control the sharding using `accelerate`. Since the policies are small in size, we use a simple DDP setup: `accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml -m train.train --config ...`.

Checkpoints are saved to `output_dir` specified in the config. This also supports automatically pushing the checkpoints to an s3 bucket when the `output_dir` starts with `s3://`.
To use this functionality, you must implement the function `common/s3.py:configure_s3(path)`, which we have left as a stub for your convenience.

## Evaluation

The recommended way to evaluate is using `eval.pipeline`, which handles checkpoint resolution, multi-seed evaluation, and result aggregation:

```bash
python -m eval.pipeline ./outputs/my_experiment \
    configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --checkpoints last \
    --datasets gsm8k,math \
    --temperatures 1.0 \
    --seeds 42,43,44 \
    --sampling_mode bernoulli-argmax \
    --save_path ./eval_results
```

Key arguments:
- First positional arg: path to directory containing `checkpoint-*` subdirectories
- Second positional arg: path to experiment config
- `--checkpoints`: comma-separated checkpoint numbers, or `first`/`last` for automatic resolution
- `--datasets`: comma-separated list, or `all` for gsm8k,math,humaneval,mbpp
- `--seeds`: comma-separated random seeds for multiple evaluation runs

Results are saved to `--save_path` as JSON files containing generations, with aggregated metrics in CSV format.

### Direct Evaluation

For more control, you can use `eval.eval` directly:

```bash
python -m eval.eval \
    --policy_path ./outputs/my_experiment/checkpoint-1000/model.safetensors \
    --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --dataset gsm8k \
    --seed 42 \
    --temperature_policy 1.0 \
    --sampling_mode bernoulli-argmax \
    --output_dir ./eval_results
```
