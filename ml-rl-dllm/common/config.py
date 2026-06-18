#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
from trl.trainer.grpo_config import GRPOConfig


@dataclass
class Config(GRPOConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.reward_type == "task_only":
            self.reward_budget_mode = "none"
        if self.loglikelihood_dtype is not None:
            assert isinstance(self.loglikelihood_dtype, str)
            if self.loglikelihood_dtype.lower() in {"none", "null"}:
                # should be handled by super already but let's manually catch it just in case it slipped by
                self.loglikelihood_dtype = None
            else:
                try:
                    self.loglikelihood_dtype = getattr(torch, self.loglikelihood_dtype)
                    if not self.loglikelihood_dtype.is_floating_point:
                        raise AttributeError()
                except AttributeError:
                    raise TypeError(
                        f"loglikelihood_dtype = {self.loglikelihood_dtype} is not a valid floating-point torch dtype"
                    )

    # Parameters that control the data preprocessing
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to the base diffusion model."},
    )
    gpu: Optional[str] = field(
        default=None,
        metadata={
            "help": "Convenience alias for cuda_visible_devices. Set to a physical "
            "GPU index such as '2' to expose only that GPU as process cuda:0. "
            "Can also be a comma-separated list for multi-GPU runs."
        },
    )
    cuda_visible_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional physical CUDA device IDs to expose before torch is initialized. "
            "For example, set to '1' to use nvidia-smi GPU index 1 as process cuda:0. "
            "Takes precedence over gpu and cuda_min_memory_gb."
        },
    )
    cuda_min_memory_gb: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional minimum total GPU memory in GiB. When CUDA_VISIBLE_DEVICES "
            "and manual GPU settings are unset, entrypoint scripts select the smallest "
            "sufficient nvidia-smi GPU before torch is initialized."
        },
    )

    # Diffusion-specific parameters

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size for generation. If `None`, defaults to effective training batch size."
        },
    )

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    temperature: float = field(
        default=0.0,
        metadata={
            "help": "Temperature for Gumbel noise during generation. 0.0 means no noise."
        },
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    reward_functions: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of reward function names to use. If None, uses dataset-specific defaults. "
            "See reward_func.py module for available functions."
        },
    )

    policy_type: str = field(
        default="dit_confidence",
        metadata={
            "help": "Type of policy to use. Options: ['dit_hidden', 'dit_confidence']."
        },
    )

    policy_num_heads: int = field(
        default=2,
        metadata={"help": "Number of attention heads for policy transformer blocks."},
    )
    policy_hidden_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hidden dimension for policy transformer blocks. If None, uses model's hidden size."
        },
    )
    policy_feedforward_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Feedforward dimension for policy transformer blocks. If None, uses 4 * hidden_dim."
        },
    )
    policy_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for policy transformer blocks."},
    )
    policy_time_embed_dim: int = field(
        default=256, metadata={"help": "Time embedding dimension for policy adaLN."}
    )
    policy_time_period: float = field(
        default=1,
        metadata={"help": "Max period for sinusoidal time embedding in policy."},
    )
    policy_num_blocks: int = field(
        default=1,
        metadata={"help": "Number of transformer blocks for policy architectures."},
    )

    policy_smart_init: float | None = field(
        default=None,
        metadata={
            "help": "Target mean for smart initialization of policy. "
            "Sets ada ln to identity and output_proj bias to this value, centering initial logits "
            "at the specified target. For DPLS sampling, use 0.0 to match stop_logit "
            "for balanced sampling, or negative values to bias toward stopping earlier. "
            "If None, uses default PyTorch initialization."
        },
    )
    policy_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional checkpoint path for loading lightweight unmasking policy weights only."
        },
    )
    policy_checkpoint_path_effective: Optional[str] = field(
        default=None,
        metadata={"help": "Resolved lightweight policy checkpoint file path."},
    )
    policy_checkpoint_loaded: bool = field(
        default=False,
        metadata={"help": "Whether lightweight policy checkpoint weights were loaded."},
    )
    policy_checkpoint_missing_keys: list[str] = field(
        default_factory=list,
        metadata={"help": "Missing keys reported while loading policy checkpoint."},
    )
    policy_checkpoint_unexpected_keys: list[str] = field(
        default_factory=list,
        metadata={"help": "Unexpected keys reported while loading policy checkpoint."},
    )

    confidences_top_p: int = field(
        default=1,
        metadata={
            "help": "The number of top confidences to use as input to the policy. Only used for dit_confidence policy type."
        },
    )

    alpha_compute_reward: float = field(
        default=0.0,
        metadata={"help": "Weight of the compute term in the reward function."},
    )

    alpha_correctness_reward: float = field(
        default=1.0,
        metadata={"help": "Weight of the correctness reward function."},
    )

    sampling_mode: str = field(
        default="bernoulli",
        metadata={
            "help": "Type of sampling strategy to use. Options: ['bernoulli', 'bernoulli-argmax', 'dpls']."
        },
    )

    dpls_stop_logit: float = field(
        default=0.0,
        metadata={
            "help": "The stop logit (utility) for DPLS sampling. Used if `sampling_mode == 'dpls'`."
        },
    )

    policy_full_context: bool = field(
        default=True,
        metadata={
            "help": "Whether to pass full sequence context to policy instead of just current block. "
            "When True, policy sees entire sequence but can only affect current block."
        },
    )

    timestep_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The batch size to use for the inner-loop policy call and log-likelihood calculations. "
            "If None, the full batch of timesteps will be processed in parallel."
        },
    )

    loglikelihood_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "If not None, loglikelihood computations and results will be in this dtype "
            "(which should be a torch floating point type such as 'float32')."
            " If None, input dtypes will be preserved (eg. bfloat16 for low-precision training)."
        },
    )

    save_best_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether to save the best checkpoint."},
    )
    analysis_only: bool = field(
        default=False,
        metadata={"help": "Run analysis/evaluation collection without optimizer steps."},
    )
    freeze_policy: bool = field(
        default=False,
        metadata={"help": "Freeze lightweight policy parameters for analysis-only runs."},
    )

    es_thresholds: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Thresholds to use for expert steering (ES) rollouts."},
    )

    # Verifier-aware budget-conditioned decoding additions.
    method: str = field(
        default="policy",
        metadata={
            "help": "High-level method label for logging and analysis. "
            "Examples: random, fixed_ratio, high_confidence, fast_dllm, learned_policy, "
            "verifier_remask, budget_policy, verifier_policy."
        },
    )
    enable_budget_conditioning: bool = field(
        default=False,
        metadata={"help": "Append target/remaining-budget features to policy inputs."},
    )
    target_budget: int = field(
        default=32,
        metadata={"help": "Target NFE budget for budget-conditioned decoding."},
    )
    target_budgets: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128],
        metadata={"help": "Budgets to evaluate for one budget-conditioned policy."},
    )
    train_budget_sampling: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128],
        metadata={"help": "Candidate target budgets for budget-conditioned training."},
    )
    train_rollout_compute_styles: list[str] = field(
        default_factory=lambda: ["normal"],
        metadata={
            "help": "Per-rollout compute styles, e.g. ['normal', 'patient']."
        },
    )
    hard_stop_at_target_budget: bool = field(
        default=True,
        metadata={
            "help": "When true, target_budget is an enforced generation cap. "
            "When false, target_budget is only a reward target/policy feature."
        },
    )
    hard_generation_budget: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional hard generation cap used when hard_stop_at_target_budget=False."
        },
    )
    patient_unmask_logit_bias: float = field(
        default=0.0,
        metadata={"help": "Logit bias applied to patient rollout unmask decisions."},
    )
    patient_policy_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature applied to patient rollout policy logits."},
    )
    patient_max_unmask_fraction: float = field(
        default=1.0,
        metadata={"help": "Maximum fraction of masked tokens patient rollout can unmask per step."},
    )
    patient_min_steps_frac: float = field(
        default=0.0,
        metadata={"help": "Minimum target-budget fraction before patient rollout can finish."},
    )
    patient_global_slowdown: bool = field(
        default=True,
        metadata={"help": "Apply global patient logit bias, temperature, and unmask caps."},
    )
    patient_delay_low_confidence: bool = field(
        default=False,
        metadata={"help": "Delay the lowest-confidence selected patient tokens."},
    )
    patient_low_confidence_quantile: float = field(
        default=0.3,
        metadata={"help": "Fraction of selected patient tokens to delay by confidence."},
    )
    patient_delay_final_window: bool = field(
        default=False,
        metadata={"help": "Delay final-window patient tokens until a minimum step."},
    )
    patient_final_window_tokens: int = field(
        default=32,
        metadata={"help": "Number of generated-token positions in the final window."},
    )
    patient_final_window_min_steps_frac: float = field(
        default=0.6,
        metadata={"help": "Target-budget fraction before final-window patient tokens can unmask."},
    )
    enable_deliberation_probe: bool = field(
        default=False,
        metadata={"help": "Save/evaluate an intermediate generation snapshot."},
    )
    deliberation_probe_frac: float = field(
        default=0.5,
        metadata={"help": "Target-budget fraction used for the deliberation probe."},
    )
    deliberation_bonus_weight: float = field(
        default=0.0,
        metadata={"help": "Reward weight for quality gain after the deliberation probe."},
    )
    patient_win_bonus: float = field(
        default=0.0,
        metadata={"help": "Reward bonus when patient rollout beats normal rollout."},
    )
    patient_win_margin: float = field(
        default=0.05,
        metadata={"help": "Minimum task-score margin for patient win bonus."},
    )

    enable_verifier: bool = field(
        default=False,
        metadata={"help": "Enable label-free verifier calls during decoding/eval."},
    )
    verifier_type: str = field(
        default="math",
        metadata={"help": "Verifier implementation to use. Currently supports 'math'."},
    )
    verifier_schedule: str = field(
        default="final_only",
        metadata={
            "help": "Verifier schedule: none, final_only, after_answer_detected, "
            "every_k_steps, final_block_only."
        },
    )
    verifier_every_k_steps: int = field(
        default=8,
        metadata={"help": "Verifier period when verifier_schedule='every_k_steps'."},
    )
    verifier_threshold: float = field(
        default=0.7,
        metadata={"help": "VerifierAnswerRemask triggers below this score."},
    )

    enable_remasking: bool = field(
        default=False,
        metadata={"help": "Enable answer-span remasking during generation."},
    )
    remask_strategy: str = field(
        default="none",
        metadata={"help": "Remasking strategy: none, low_confidence, verifier_answer."},
    )
    max_remasks_per_sample: int = field(
        default=1,
        metadata={"help": "Maximum remask actions per sample."},
    )
    remask_cooldown_steps: int = field(
        default=2,
        metadata={"help": "Minimum steps between remasks for a sample."},
    )
    min_steps_before_remask: int = field(
        default=0,
        metadata={"help": "Do not remask before this many denoising steps."},
    )
    remask_answer_span_mode: str = field(
        default="numeric_only",
        metadata={"help": "Answer span mode for remasking. Default: numeric_only."},
    )
    remask_penalty: float = field(
        default=0.1,
        metadata={"help": "Diagnostic/default remask penalty coefficient."},
    )

    reward_type: str = field(
        default="multiplicative_budget",
        metadata={
            "help": "Config-selected reward: additive_proposal, multiplicative_budget, or task_only."
        },
    )
    reward_gsm_extractor: str = field(
        default="robust",
        metadata={"help": "GSM8K reward extractor: strict, answer_span, or robust."},
    )
    reward_quality_mode: str = field(
        default="none",
        metadata={
            "help": "Reward quality mode: none, format_weighted, or "
            "source_aware_anti_junk."
        },
    )
    answer_span_partial_credit: float = field(
        default=0.4,
        metadata={"help": "Task-score credit for answer-span-only correct answers."},
    )
    malformed_answer_factor: float = field(
        default=0.5,
        metadata={"help": "Quality multiplier for non-strict answer formats."},
    )
    source_answer_marker_credit: float = field(
        default=0.20,
        metadata={"help": "Source-aware reward credit for answer-marker answers."},
    )
    source_final_line_credit: float = field(
        default=0.15,
        metadata={"help": "Source-aware reward credit for final-line answers."},
    )
    source_final_window_credit: float = field(
        default=0.08,
        metadata={"help": "Source-aware reward credit for final-window answers."},
    )
    source_incomplete_answer_tag_credit: float = field(
        default=0.03,
        metadata={"help": "Source-aware reward credit for incomplete answer tags."},
    )
    source_other_answer_span_credit: float = field(
        default=0.05,
        metadata={"help": "Source-aware reward credit for other answer-span sources."},
    )
    enable_repetition_penalty: bool = field(
        default=False,
        metadata={"help": "Enable repetitive-output reward penalty."},
    )
    repetition_min_tokens: int = field(
        default=8,
        metadata={"help": "Minimum token count for ratio-based repetition checks."},
    )
    repetition_number_run_threshold: int = field(
        default=3,
        metadata={"help": "Repeated normalized-number threshold for junk detection."},
    )
    repetition_token_run_threshold: int = field(
        default=4,
        metadata={"help": "Consecutive repeated-token threshold for junk detection."},
    )
    repetition_bigram_run_threshold: int = field(
        default=3,
        metadata={"help": "Repeated bigram threshold for junk detection."},
    )
    repetition_answer_span_number_threshold: int = field(
        default=2,
        metadata={
            "help": "Repeated normalized-number threshold inside answer tags."
        },
    )
    repetition_unique_ratio_threshold: float = field(
        default=0.25,
        metadata={"help": "Unique-token ratio threshold for repetition penalty."},
    )
    repetition_max_freq_threshold: float = field(
        default=0.25,
        metadata={"help": "Max token frequency threshold for repetition penalty."},
    )
    repetition_penalty_factor: float = field(
        default=0.5,
        metadata={"help": "Multiplicative factor applied for each repetition signal."},
    )
    zero_reward_if_repeated_answer_span: bool = field(
        default=True,
        metadata={"help": "Zero source-aware task score for repeated answer spans."},
    )
    max_reward_if_malformed: Optional[float] = field(
        default=None,
        metadata={"help": "Optional final reward cap for malformed positive answers."},
    )
    max_reward_if_repetitive: Optional[float] = field(
        default=None,
        metadata={"help": "Optional final reward cap for repetitive positive answers."},
    )
    enable_negative_junk_penalty: bool = field(
        default=False,
        metadata={"help": "Subtract a small negative reward for wrong junk outputs."},
    )
    repeated_junk_negative_reward: float = field(
        default=0.02,
        metadata={"help": "Negative reward subtracted for repetitive wrong output."},
    )
    malformed_junk_negative_reward: float = field(
        default=0.01,
        metadata={"help": "Negative reward subtracted for malformed wrong output."},
    )
    incomplete_answer_tag_negative_reward: float = field(
        default=0.01,
        metadata={
            "help": "Negative reward subtracted for incomplete answer-tag outputs."
        },
    )
    min_reward_floor: float = field(
        default=-0.05,
        metadata={"help": "Lower bound after optional negative junk penalties."},
    )
    reward_budget_mode: str = field(
        default="cap",
        metadata={"help": "Budget reward mode: none, cap, band, or threshold."},
    )
    threshold_reward_hard_zero: bool = field(
        default=False,
        metadata={"help": "If true, threshold reward is zero when nfe_used exceeds target_budget."},
    )
    reward_lambda_compute: float = field(
        default=0.01,
        metadata={"help": "Compute penalty for additive_proposal reward."},
    )
    reward_normalize_compute: bool = field(
        default=True,
        metadata={"help": "Normalize NFE by max_steps in additive_proposal reward."},
    )
    reward_beta: float = field(
        default=2.0,
        metadata={"help": "Over-target-budget penalty for multiplicative reward."},
    )
    reward_mu: float = field(
        default=0.5,
        metadata={"help": "General NFE penalty for multiplicative reward."},
    )
    reward_rho: float = field(
        default=0.1,
        metadata={"help": "Remask-count penalty for multiplicative reward."},
    )

    enable_posthoc_continuation: bool = field(
        default=False,
        metadata={"help": "Evaluate verifier-guided post-hoc selective continuation candidates."},
    )
    continuation_steps: list[int] = field(
        default_factory=list,
        metadata={"help": "Extra denoising step counts to evaluate for post-hoc continuation."},
    )
    continuation_trigger: str = field(
        default="heuristic",
        metadata={"help": "Continuation trigger: always, answer_span, low_confidence, verifier_bad, answer_span_or_low_confidence, heuristic."},
    )
    continuation_remask_mode: str = field(
        default="answer_span",
        metadata={"help": "Continuation remask mode: answer_span, final_window, low_confidence_answer_window."},
    )
    continuation_final_window_tokens: int = field(
        default=32,
        metadata={"help": "Final generated-token window size for continuation remasking."},
    )
    continuation_low_confidence_quantile: float = field(
        default=0.3,
        metadata={"help": "Lowest-confidence fraction to remask inside the continuation answer/final window."},
    )
    continuation_min_remaining_budget: int = field(
        default=1,
        metadata={"help": "Minimum remaining target budget required for non-oracle continuation."},
    )
    continuation_use_oracle_for_analysis: bool = field(
        default=False,
        metadata={"help": "Evaluate continuation candidates even when they exceed the target budget."},
    )
    continuation_force_for_analysis: bool = field(
        default=False,
        metadata={"help": "Force continuation candidate evaluation for analysis data collection."},
    )
    continuation_select_best_k_oracle: bool = field(
        default=False,
        metadata={"help": "Log best continuation candidate by oracle task score during analysis."},
    )
    continuation_confidence_threshold: float = field(
        default=0.5,
        metadata={"help": "Answer-span confidence threshold for continuation triggers."},
    )
    continuation_verifier_threshold: float = field(
        default=0.7,
        metadata={"help": "Verifier-score threshold for continuation triggers."},
    )

    log_step_traces: bool = field(
        default=False,
        metadata={"help": "Save per-step decoding traces when eval output is enabled."},
    )
    save_generations: bool = field(
        default=True,
        metadata={"help": "Save generation records during evaluation."},
    )
    save_continuation_records: bool = field(
        default=False,
        metadata={"help": "Save flattened post-hoc continuation candidate records."},
    )
    save_verifier_results: bool = field(
        default=True,
        metadata={"help": "Save verifier result records during evaluation."},
    )
    count_verifier_time: bool = field(
        default=True,
        metadata={"help": "Include verifier calls in wall-clock timing measurements."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional deterministic cap on shuffled training examples."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional deterministic cap on shuffled evaluation examples."
        },
    )
