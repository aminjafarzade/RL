#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import gc
import json
import os
import warnings
from collections import deque
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import torch
import wandb
from accelerate.utils import gather
from accelerate.utils import gather_object
from datasets import Dataset
from datasets import IterableDataset
from torch import nn
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback
from trl.data_utils import is_conversational
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample

from common.budgeting import assign_rollout_compute_styles
from common.budgeting import sample_group_target_budgets
from common.generation.generation import generate_unified
from common.generation.sampling import bernoulli_batch_loglik
from common.generation.sampling import dpls_batch_loglik
from common.policy_features import get_policy_extra_feature_names
from common.rewards import apply_reward_quality_caps
from common.rewards import apply_negative_junk_penalty
from common.rewards import compute_budget_reward
from common.rewards import compute_gsm_reward_parse_fields
from common.rewards import compute_reward_quality
from common.rewards import deliberation_bonus
from common.rewards import has_malformed_answer_structure
from common.rewards import patient_win_bonus_applies
from common.s3 import S3UploadCallback

try:
    import rich  # noqa: F401

    _rich_available = True
except ImportError:
    _rich_available = False

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _effective_reward_budget_mode(args) -> str:
    if getattr(args, "reward_type", None) == "task_only":
        return "none"
    return getattr(args, "reward_budget_mode", "cap")


def _uses_task_only_quality_scale(args) -> bool:
    return _effective_reward_budget_mode(args) == "none"


def _reward_quality_kwargs_from_args(args) -> dict[str, Any]:
    return {
        "extractor_mode": args.reward_gsm_extractor,
        "reward_quality_mode": args.reward_quality_mode,
        "answer_span_partial_credit": args.answer_span_partial_credit,
        "malformed_answer_factor": args.malformed_answer_factor,
        "source_answer_marker_credit": args.source_answer_marker_credit,
        "source_final_line_credit": args.source_final_line_credit,
        "source_final_window_credit": args.source_final_window_credit,
        "source_incomplete_answer_tag_credit": (
            args.source_incomplete_answer_tag_credit
        ),
        "source_other_answer_span_credit": args.source_other_answer_span_credit,
        "enable_repetition_penalty": args.enable_repetition_penalty,
        "repetition_min_tokens": args.repetition_min_tokens,
        "repetition_number_run_threshold": args.repetition_number_run_threshold,
        "repetition_token_run_threshold": args.repetition_token_run_threshold,
        "repetition_bigram_run_threshold": args.repetition_bigram_run_threshold,
        "repetition_answer_span_number_threshold": (
            args.repetition_answer_span_number_threshold
        ),
        "repetition_unique_ratio_threshold": args.repetition_unique_ratio_threshold,
        "repetition_max_freq_threshold": args.repetition_max_freq_threshold,
        "repetition_penalty_factor": args.repetition_penalty_factor,
        "zero_reward_if_repeated_answer_span": (
            args.zero_reward_if_repeated_answer_span
        ),
    }


def _gsm_reward_parse_fields(
    generated_text: str | None,
    reference_answer,
    args,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return compute_gsm_reward_parse_fields(
        generated_text,
        reference_answer,
        args,
        metadata,
    )


def _visible_token_count(tokenizer, text: str | None) -> int | None:
    if tokenizer is None or text is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


class Trainer(GRPOTrainer):
    def __init__(
        self,
        model,
        dllm: nn.Module,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (
            None,
            None,
        ),
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=None,
        )
        torch._dynamo.config.capture_scalar_outputs = True
        self.dllm = torch.compile(dllm)
        self.dllm.eval()

        # Initialize buffering for multi-iteration training
        self._buffered_inputs = None
        self._step = 0

        if self.args.remasking == "policy":
            assert self.beta == 0.0, "Beta must be 0.0 for policy-based remasking"

        # Gradient accumulation not supported with current buffering logic
        assert self.args.gradient_accumulation_steps == 1, (
            "gradient_accumulation_steps must be 1 (current buffering does not support gradient accumulation)"
        )

        # Track recent rewards for best checkpoint saving
        self.train_reward_queue = deque(
            maxlen=10 * self.args.gradient_accumulation_steps
        )
        self.train_reward_best = -float("inf")
        self.train_reward_best_step = 0
        self.effective_steps = 0
        self.s3_callback = None
        for callback in callbacks:
            if isinstance(callback, S3UploadCallback):
                self.s3_callback = callback
                break

    def _write_generation_records(
        self,
        *,
        mode: str,
        inputs: list[dict[str, Any]],
        prompts_text: list[str],
        completions_text: list[str],
        metadata: list[dict[str, Any]],
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        if not (self.args.save_generations or self.args.save_verifier_results):
            return

        os.makedirs(self.args.output_dir, exist_ok=True)
        reward_names = [
            reward_func.__name__
            if not isinstance(reward_func, nn.Module)
            else reward_func.config._name_or_path.split("/")[-1]
            for reward_func in self.reward_funcs
        ]
        rewards_list = rewards.detach().cpu().tolist()
        rewards_per_func_list = rewards_per_func.detach().cpu().tolist()
        effective_reward_budget_mode = _effective_reward_budget_mode(self.args)
        task_only_quality_scale = _uses_task_only_quality_scale(self.args)
        task_scores_for_pairs: list[float | None] = []
        quality_for_caps: list[dict[str, Any] | None] = []
        uncapped_rewards: list[float | None] = []
        threshold_gates_for_caps: list[float | None] = []
        for idx, example in enumerate(inputs):
            item_metadata = metadata[idx] if idx < len(metadata) else {}
            generated_text = (
                completions_text[idx]
                if idx < len(completions_text)
                else item_metadata.get("generated_text_final")
            )
            if example.get("dataset_type", self.args.dataset) == "gsm8k":
                quality = compute_reward_quality(
                    generated_text,
                    example.get("answer"),
                    format_ok=item_metadata.get("format_ok"),
                    **_reward_quality_kwargs_from_args(self.args),
                )
                task_score = quality.task_score
                if not task_only_quality_scale:
                    task_score *= float(self.args.alpha_correctness_reward)
                is_clean_format = bool(quality.strict_correct) or bool(
                    quality.format_ok
                )
                repeated_junk_detected = bool(quality.repeated_junk_detected) or (
                    float(quality.repetition_penalty) < 1.0
                )
                if not task_only_quality_scale:
                    task_score_cap = apply_reward_quality_caps(
                        reward=task_score,
                        is_malformed=not is_clean_format,
                        repeated_junk_detected=repeated_junk_detected,
                        max_reward_if_malformed=self.args.max_reward_if_malformed,
                        max_reward_if_repetitive=self.args.max_reward_if_repetitive,
                    )
                    task_score = task_score_cap.reward_after_quality_caps
                task_scores_for_pairs.append(task_score)
                budget = compute_budget_reward(
                    task_score=task_score,
                    nfe_used=item_metadata.get(
                        "actual_nfe",
                        item_metadata.get("nfe_used", 0),
                    ),
                    target_budget=item_metadata.get(
                        "target_budget",
                        getattr(self.args, "target_budget", 1),
                    ),
                    max_steps=item_metadata.get(
                        "max_steps",
                        getattr(self.args, "max_completion_length", 1),
                    ),
                    remask_count=item_metadata.get("remask_count", 0),
                    reward_budget_mode=effective_reward_budget_mode,
                    reward_beta=self.args.reward_beta,
                    reward_mu=self.args.reward_mu,
                    reward_rho=self.args.reward_rho,
                    threshold_reward_hard_zero=self.args.threshold_reward_hard_zero,
                )
                early_score = 0.0
                early_text = item_metadata.get("early_generated_text")
                if early_text is not None:
                    early_score = compute_reward_quality(
                        early_text,
                        example.get("answer"),
                        format_ok=None,
                        **_reward_quality_kwargs_from_args(self.args),
                    ).task_score
                    if not task_only_quality_scale:
                        early_score *= float(self.args.alpha_correctness_reward)
                _, probe_bonus = deliberation_bonus(
                    early_task_score=early_score,
                    final_task_score=task_score,
                    threshold_gate=budget.threshold_gate,
                    deliberation_bonus_weight=self.args.deliberation_bonus_weight,
                )
                reward_before_caps = budget.reward
                if (
                    self.args.enable_deliberation_probe
                    and effective_reward_budget_mode != "none"
                ):
                    reward_before_caps += probe_bonus
                quality_for_caps.append(
                    {
                        "strict_correct": quality.strict_correct,
                        "format_ok": item_metadata.get("format_ok"),
                        "repetition_penalty": quality.repetition_penalty,
                        "repeated_char_run_detected": (
                            quality.repeated_char_run_detected
                        ),
                        "repeated_junk_detected": quality.repeated_junk_detected,
                        "answer_source": quality.answer_source,
                        "has_malformed_answer_structure": (
                            has_malformed_answer_structure(generated_text)
                        ),
                    }
                )
                uncapped_rewards.append(float(reward_before_caps))
                threshold_gates_for_caps.append(budget.threshold_gate)
            else:
                task_scores_for_pairs.append(None)
                quality_for_caps.append(None)
                uncapped_rewards.append(None)
                threshold_gates_for_caps.append(None)

        pair_statuses: dict[int, dict[str, bool]] = {}
        grouped_indices: dict[Any, list[int]] = {}
        for idx, item_metadata in enumerate(metadata):
            group_id = item_metadata.get("group_id")
            if group_id is not None:
                grouped_indices.setdefault(group_id, []).append(idx)
        for indices in grouped_indices.values():
            normal_idx = next(
                (
                    idx
                    for idx in indices
                    if metadata[idx].get("rollout_compute_style") == "normal"
                ),
                None,
            )
            patient_idx = next(
                (
                    idx
                    for idx in indices
                    if metadata[idx].get("rollout_compute_style") == "patient"
                ),
                None,
            )
            if normal_idx is None or patient_idx is None:
                continue
            normal_score = task_scores_for_pairs[normal_idx]
            patient_score = task_scores_for_pairs[patient_idx]
            if normal_score is None or patient_score is None:
                continue
            if (
                effective_reward_budget_mode != "none"
                and self.args.patient_win_bonus > 0.0
                and uncapped_rewards[patient_idx] is not None
                and patient_win_bonus_applies(
                    patient_task_score=patient_score,
                    normal_task_score=normal_score,
                    patient_nfe=metadata[patient_idx].get(
                        "actual_nfe",
                        metadata[patient_idx].get("nfe_used", 0),
                    ),
                    target_budget=metadata[patient_idx].get(
                        "target_budget",
                        getattr(self.args, "target_budget", 1),
                    ),
                    patient_win_margin=self.args.patient_win_margin,
                )
            ):
                uncapped_rewards[patient_idx] += (
                    self.args.patient_win_bonus
                    * float(threshold_gates_for_caps[patient_idx] or 0.0)
                )
            if mode == "train":
                status = {
                    "patient_correct_normal_wrong": (
                        patient_score > 0.0 and normal_score <= 0.0
                    ),
                    "normal_correct_patient_wrong": (
                        normal_score > 0.0 and patient_score <= 0.0
                    ),
                    "patient_better_than_normal": (
                        patient_score > normal_score + self.args.patient_win_margin
                    ),
                    "both_good": normal_score > 0.0 and patient_score > 0.0,
                    "both_bad": normal_score <= 0.0 and patient_score <= 0.0,
                }
                pair_statuses[normal_idx] = status
                pair_statuses[patient_idx] = status

        cap_fields_by_idx: dict[int, dict[str, Any]] = {}
        for idx, reward_before_caps in enumerate(uncapped_rewards):
            quality = quality_for_caps[idx]
            if reward_before_caps is None or quality is None:
                continue
            if task_only_quality_scale:
                repeated_junk_detected = bool(quality["repeated_junk_detected"]) or (
                    float(quality["repetition_penalty"]) < 1.0
                )
                negative = apply_negative_junk_penalty(
                    reward=reward_before_caps,
                    strict_correct=bool(quality["strict_correct"]),
                    repeated_junk_detected=repeated_junk_detected,
                    malformed_junk_detected=bool(
                        quality["has_malformed_answer_structure"]
                    ),
                    answer_source=quality.get("answer_source"),
                    enable_negative_junk_penalty=(
                        self.args.enable_negative_junk_penalty
                    ),
                    repeated_junk_negative_reward=(
                        self.args.repeated_junk_negative_reward
                    ),
                    malformed_junk_negative_reward=(
                        self.args.malformed_junk_negative_reward
                    ),
                    incomplete_answer_tag_negative_reward=(
                        self.args.incomplete_answer_tag_negative_reward
                    ),
                    min_reward_floor=self.args.min_reward_floor,
                )
                cap_fields_by_idx[idx] = {
                    "max_reward_if_malformed": self.args.max_reward_if_malformed,
                    "max_reward_if_repetitive": self.args.max_reward_if_repetitive,
                    "reward_before_quality_caps": reward_before_caps,
                    "reward_after_quality_caps": reward_before_caps,
                    "malformed_reward_cap_applied": False,
                    "repetitive_reward_cap_applied": False,
                    "malformed_capped": False,
                    "repetitive_capped": False,
                    "enable_negative_junk_penalty": (
                        self.args.enable_negative_junk_penalty
                    ),
                    "repeated_junk_negative_reward": (
                        self.args.repeated_junk_negative_reward
                    ),
                    "malformed_junk_negative_reward": (
                        self.args.malformed_junk_negative_reward
                    ),
                    "incomplete_answer_tag_negative_reward": (
                        self.args.incomplete_answer_tag_negative_reward
                    ),
                    "min_reward_floor": self.args.min_reward_floor,
                    "reward_before_negative_junk_penalty": (
                        negative.reward_before_negative_junk_penalty
                    ),
                    "reward_after_negative_junk_penalty": (
                        negative.reward_after_negative_junk_penalty
                    ),
                    "negative_junk_penalty": negative.negative_junk_penalty,
                    "repeated_junk_negative_penalty_applied": (
                        negative.repeated_junk_negative_penalty_applied
                    ),
                    "malformed_junk_negative_penalty_applied": (
                        negative.malformed_junk_negative_penalty_applied
                    ),
                    "incomplete_answer_tag_negative_penalty_applied": (
                        negative.incomplete_answer_tag_negative_penalty_applied
                    ),
                }
                continue
            is_clean_format = bool(quality["strict_correct"]) or bool(
                quality["format_ok"]
            )
            repeated_junk_detected = bool(quality["repeated_junk_detected"]) or (
                float(quality["repetition_penalty"]) < 1.0
            )
            cap = apply_reward_quality_caps(
                reward=reward_before_caps,
                is_malformed=not is_clean_format,
                repeated_junk_detected=repeated_junk_detected,
                max_reward_if_malformed=self.args.max_reward_if_malformed,
                max_reward_if_repetitive=self.args.max_reward_if_repetitive,
            )
            cap_fields_by_idx[idx] = {
                "max_reward_if_malformed": self.args.max_reward_if_malformed,
                "max_reward_if_repetitive": self.args.max_reward_if_repetitive,
                "reward_before_quality_caps": (
                    cap.reward_before_quality_caps
                ),
                "reward_after_quality_caps": cap.reward_after_quality_caps,
                "malformed_reward_cap_applied": (
                    cap.malformed_reward_cap_applied
                ),
                "repetitive_reward_cap_applied": (
                    cap.repetitive_reward_cap_applied
                ),
                "malformed_capped": cap.malformed_reward_cap_applied,
                "repetitive_capped": cap.repetitive_reward_cap_applied,
            }
            negative = apply_negative_junk_penalty(
                reward=cap.reward_after_quality_caps,
                strict_correct=bool(quality["strict_correct"]),
                repeated_junk_detected=repeated_junk_detected,
                malformed_junk_detected=bool(
                    quality["has_malformed_answer_structure"]
                ),
                answer_source=quality.get("answer_source"),
                enable_negative_junk_penalty=self.args.enable_negative_junk_penalty,
                repeated_junk_negative_reward=(
                    self.args.repeated_junk_negative_reward
                ),
                malformed_junk_negative_reward=(
                    self.args.malformed_junk_negative_reward
                ),
                incomplete_answer_tag_negative_reward=(
                    self.args.incomplete_answer_tag_negative_reward
                ),
                min_reward_floor=self.args.min_reward_floor,
            )
            cap_fields_by_idx[idx].update(
                {
                    "enable_negative_junk_penalty": (
                        self.args.enable_negative_junk_penalty
                    ),
                    "repeated_junk_negative_reward": (
                        self.args.repeated_junk_negative_reward
                    ),
                    "malformed_junk_negative_reward": (
                        self.args.malformed_junk_negative_reward
                    ),
                    "incomplete_answer_tag_negative_reward": (
                        self.args.incomplete_answer_tag_negative_reward
                    ),
                    "min_reward_floor": self.args.min_reward_floor,
                    "reward_before_negative_junk_penalty": (
                        negative.reward_before_negative_junk_penalty
                    ),
                    "reward_after_negative_junk_penalty": (
                        negative.reward_after_negative_junk_penalty
                    ),
                    "negative_junk_penalty": negative.negative_junk_penalty,
                    "repeated_junk_negative_penalty_applied": (
                        negative.repeated_junk_negative_penalty_applied
                    ),
                    "malformed_junk_negative_penalty_applied": (
                        negative.malformed_junk_negative_penalty_applied
                    ),
                    "incomplete_answer_tag_negative_penalty_applied": (
                        negative.incomplete_answer_tag_negative_penalty_applied
                    ),
                }
            )

        per_example_path = os.path.join(self.args.output_dir, "per_example.jsonl")
        generations_path = os.path.join(self.args.output_dir, "generations.jsonl")
        verifier_path = os.path.join(self.args.output_dir, "verifier_results.jsonl")
        eval_samples_path = os.path.join(self.args.output_dir, "eval_samples.jsonl")

        with open(per_example_path, "a") as per_example_file:
            generations_file = (
                open(generations_path, "a") if self.args.save_generations else None
            )
            verifier_file = (
                open(verifier_path, "a") if self.args.save_verifier_results else None
            )
            eval_samples_file = (
                open(eval_samples_path, "a") if mode == "eval" else None
            )
            try:
                for idx, example in enumerate(inputs):
                    item_metadata = metadata[idx] if idx < len(metadata) else {}
                    generated_text = (
                        completions_text[idx]
                        if idx < len(completions_text)
                        else item_metadata.get("generated_text_final")
                    )
                    reference_answer = example.get("answer")
                    dataset_name = example.get("dataset_type", self.args.dataset)
                    gsm_fields = (
                        _gsm_reward_parse_fields(
                            generated_text,
                            reference_answer,
                            self.args,
                            item_metadata,
                        )
                        if dataset_name == "gsm8k"
                        else {}
                    )
                    visible_generated_char_count = len(generated_text or "")
                    visible_generated_token_count = _visible_token_count(
                        self.processing_class,
                        generated_text,
                    )
                    reward_components = {
                        reward_names[j]: rewards_per_func_list[idx][j]
                        for j in range(len(reward_names))
                    }
                    record = {
                        **item_metadata,
                        **gsm_fields,
                        "mode": mode,
                        "trainer_global_step": int(self.state.global_step),
                        "generation_step": int(self._step),
                        "example_id": f"{mode}-{self.state.global_step}-{idx}",
                        "dataset": dataset_name,
                        "prompt": prompts_text[idx] if idx < len(prompts_text) else None,
                        "reference_answer": reference_answer,
                        "generated_text": generated_text,
                        "generated_text_final": item_metadata.get(
                            "generated_text_final",
                            generated_text,
                        ),
                        "visible_generated_char_count": visible_generated_char_count,
                        "visible_generated_token_count": visible_generated_token_count,
                        "reward": rewards_list[idx],
                        "final_reward": rewards_list[idx],
                        "reward_components": reward_components,
                        **cap_fields_by_idx.get(idx, {}),
                        **pair_statuses.get(idx, {}),
                        "config_dataset": self.args.dataset,
                        "method_name": self.args.method,
                        "reward_type": self.args.reward_type,
                        "reward_type/task_only": float(
                            self.args.reward_type == "task_only"
                        ),
                        "reward_gsm_extractor": self.args.reward_gsm_extractor,
                        "reward_quality_mode": self.args.reward_quality_mode,
                        "answer_span_partial_credit": (
                            self.args.answer_span_partial_credit
                        ),
                        "malformed_answer_factor": self.args.malformed_answer_factor,
                        "source_answer_marker_credit": (
                            self.args.source_answer_marker_credit
                        ),
                        "source_final_line_credit": (
                            self.args.source_final_line_credit
                        ),
                        "source_final_window_credit": (
                            self.args.source_final_window_credit
                        ),
                        "source_incomplete_answer_tag_credit": (
                            self.args.source_incomplete_answer_tag_credit
                        ),
                        "source_other_answer_span_credit": (
                            self.args.source_other_answer_span_credit
                        ),
                        "enable_repetition_penalty": (
                            self.args.enable_repetition_penalty
                        ),
                        "repetition_min_tokens": self.args.repetition_min_tokens,
                        "repetition_number_run_threshold": (
                            self.args.repetition_number_run_threshold
                        ),
                        "repetition_token_run_threshold": (
                            self.args.repetition_token_run_threshold
                        ),
                        "repetition_bigram_run_threshold": (
                            self.args.repetition_bigram_run_threshold
                        ),
                        "repetition_answer_span_number_threshold": (
                            self.args.repetition_answer_span_number_threshold
                        ),
                        "repetition_unique_ratio_threshold": (
                            self.args.repetition_unique_ratio_threshold
                        ),
                        "repetition_max_freq_threshold": (
                            self.args.repetition_max_freq_threshold
                        ),
                        "repetition_penalty_factor": (
                            self.args.repetition_penalty_factor
                        ),
                        "zero_reward_if_repeated_answer_span": (
                            self.args.zero_reward_if_repeated_answer_span
                        ),
                        "max_reward_if_malformed": (
                            self.args.max_reward_if_malformed
                        ),
                        "max_reward_if_repetitive": (
                            self.args.max_reward_if_repetitive
                        ),
                        "enable_negative_junk_penalty": (
                            self.args.enable_negative_junk_penalty
                        ),
                        "repeated_junk_negative_reward": (
                            self.args.repeated_junk_negative_reward
                        ),
                        "malformed_junk_negative_reward": (
                            self.args.malformed_junk_negative_reward
                        ),
                        "incomplete_answer_tag_negative_reward": (
                            self.args.incomplete_answer_tag_negative_reward
                        ),
                        "min_reward_floor": self.args.min_reward_floor,
                        "reward_budget_mode": effective_reward_budget_mode,
                        "threshold_reward_hard_zero": (
                            self.args.threshold_reward_hard_zero
                        ),
                        "policy_smart_init": self.args.policy_smart_init,
                        "policy_checkpoint_path": self.args.policy_checkpoint_path,
                        "policy_checkpoint_path_requested": (
                            self.args.policy_checkpoint_path
                        ),
                        "policy_checkpoint_path_effective": (
                            self.args.policy_checkpoint_path_effective
                        ),
                        "policy_checkpoint_loaded": (
                            self.args.policy_checkpoint_loaded
                        ),
                        "policy_checkpoint_missing_keys": list(
                            self.args.policy_checkpoint_missing_keys
                        ),
                        "policy_checkpoint_unexpected_keys": list(
                            self.args.policy_checkpoint_unexpected_keys
                        ),
                        "train_rollout_compute_styles": list(
                            self.args.train_rollout_compute_styles
                        ),
                        "rollout_compute_style": item_metadata.get(
                            "rollout_compute_style",
                            "normal",
                        ),
                        "patient_unmask_logit_bias": (
                            self.args.patient_unmask_logit_bias
                        ),
                        "patient_policy_temperature": (
                            self.args.patient_policy_temperature
                        ),
                        "patient_max_unmask_fraction": (
                            self.args.patient_max_unmask_fraction
                        ),
                        "patient_min_steps_frac": self.args.patient_min_steps_frac,
                        "patient_global_slowdown": self.args.patient_global_slowdown,
                        "patient_delay_low_confidence": (
                            self.args.patient_delay_low_confidence
                        ),
                        "patient_low_confidence_quantile": (
                            self.args.patient_low_confidence_quantile
                        ),
                        "patient_delay_final_window": (
                            self.args.patient_delay_final_window
                        ),
                        "patient_final_window_tokens": (
                            self.args.patient_final_window_tokens
                        ),
                        "patient_final_window_min_steps_frac": (
                            self.args.patient_final_window_min_steps_frac
                        ),
                        "enable_deliberation_probe": (
                            self.args.enable_deliberation_probe
                        ),
                        "deliberation_probe_frac": self.args.deliberation_probe_frac,
                        "deliberation_bonus_weight": (
                            self.args.deliberation_bonus_weight
                        ),
                        "patient_win_bonus": self.args.patient_win_bonus,
                        "patient_win_margin": self.args.patient_win_margin,
                        "enable_posthoc_continuation": (
                            self.args.enable_posthoc_continuation
                        ),
                        "continuation_steps_config": list(
                            self.args.continuation_steps
                        ),
                        "continuation_trigger_config": (
                            self.args.continuation_trigger
                        ),
                        "continuation_remask_mode_config": (
                            self.args.continuation_remask_mode
                        ),
                        "continuation_final_window_tokens": (
                            self.args.continuation_final_window_tokens
                        ),
                        "continuation_low_confidence_quantile": (
                            self.args.continuation_low_confidence_quantile
                        ),
                        "continuation_min_remaining_budget": (
                            self.args.continuation_min_remaining_budget
                        ),
                        "continuation_use_oracle_for_analysis": (
                            self.args.continuation_use_oracle_for_analysis
                        ),
                        "continuation_force_for_analysis": (
                            self.args.continuation_force_for_analysis
                        ),
                        "continuation_select_best_k_oracle": (
                            self.args.continuation_select_best_k_oracle
                        ),
                        "continuation_confidence_threshold": (
                            self.args.continuation_confidence_threshold
                        ),
                        "continuation_verifier_threshold": (
                            self.args.continuation_verifier_threshold
                        ),
                        "num_generations": self.args.num_generations,
                        "max_completion_length": self.args.max_completion_length,
                        "block_length": self.args.block_length,
                        "generation_batch_size": self.args.generation_batch_size,
                        "per_device_train_batch_size": self.args.per_device_train_batch_size,
                        "target_budgets_config": list(self.args.target_budgets),
                        "train_budget_sampling_config": list(
                            self.args.train_budget_sampling
                        ),
                        "enable_budget_conditioning": self.args.enable_budget_conditioning,
                        "enable_verifier": self.args.enable_verifier,
                        "verifier_type": self.args.verifier_type,
                        "verifier_schedule": self.args.verifier_schedule,
                        "enable_remasking": self.args.enable_remasking,
                        "max_remasks_per_sample": self.args.max_remasks_per_sample,
                        "hard_stop_at_target_budget": item_metadata.get(
                            "hard_stop_at_target_budget",
                            self.args.hard_stop_at_target_budget,
                        ),
                        "configured_hard_stop_at_target_budget": (
                            self.args.hard_stop_at_target_budget
                        ),
                        "hard_generation_budget": item_metadata.get(
                            "hard_generation_budget",
                            self.args.hard_generation_budget,
                        ),
                        "reward_beta": self.args.reward_beta,
                        "reward_mu": self.args.reward_mu,
                        "reward_rho": self.args.reward_rho,
                    }
                    line = json.dumps(_jsonable(record), sort_keys=False)
                    per_example_file.write(line + "\n")
                    if generations_file is not None:
                        generations_file.write(line + "\n")
                    if verifier_file is not None:
                        verifier_record = {
                            "mode": record["mode"],
                            "trainer_global_step": record["trainer_global_step"],
                            "example_id": record["example_id"],
                            "dataset": record["dataset"],
                            "target_budget": record.get("target_budget"),
                            "nfe_used": record.get("nfe_used"),
                            "actual_nfe": record.get("actual_nfe"),
                            "verifier_calls": record.get("verifier_calls"),
                            "final_verifier_score": record.get(
                                "final_verifier_score"
                            ),
                            "parse_ok": record.get("parse_ok"),
                            "format_ok": record.get("format_ok"),
                            "arithmetic_ok": record.get("arithmetic_ok"),
                            "parsed_answer": record.get("parsed_answer"),
                            "verifier_result": record.get("verifier_result"),
                        }
                        verifier_file.write(
                            json.dumps(_jsonable(verifier_record), sort_keys=False)
                            + "\n"
                        )
                    if eval_samples_file is not None:
                        eval_record = {
                            "prompt": record.get("prompt"),
                            "reference_answer": record.get("reference_answer"),
                            "generated_text_final": record.get(
                                "generated_text_final"
                            ),
                            "strict_parsed_answer": record.get(
                                "strict_parsed_answer"
                            ),
                            "answer_span_parsed_answer": record.get(
                                "answer_span_parsed_answer"
                            ),
                            "robust_parsed_answer": record.get(
                                "robust_parsed_answer"
                            ),
                            "strict_correct": record.get("strict_correct"),
                            "answer_span_correct": record.get("answer_span_correct"),
                            "robust_correct": record.get("robust_correct"),
                            "reward_answer_source": record.get(
                                "reward_answer_source"
                            ),
                            "reward": record.get("reward"),
                            "target_budget": record.get("target_budget"),
                            "nfe_used": record.get("nfe_used"),
                        }
                        eval_samples_file.write(
                            json.dumps(_jsonable(eval_record), sort_keys=False) + "\n"
                        )
            finally:
                if generations_file is not None:
                    generations_file.close()
                if verifier_file is not None:
                    verifier_file.close()
                if eval_samples_file is not None:
                    eval_samples_file.close()

    def train(self, *args, **kwargs):
        """Override train to save final checkpoint at end of training."""
        output = super().train(*args, **kwargs)

        if self.accelerator.is_main_process:
            final_step = self.state.global_step
            checkpoint_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{final_step}"
            )

            print(f"\nSaving final checkpoint at step {final_step}")
            unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
            unwrapped_model.save_pretrained(checkpoint_dir)
            self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

            if self.s3_callback is not None:
                print(
                    f"Uploading checkpoint-{final_step} to s3: {self.args.output_dir}"
                )
                self.s3_callback.on_save(self.args, self.state, self.control)

        return output

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Override training_step to skip optimizer step when advantages are zero."""
        # Check if all advantages are zero (no learning signal)
        if "advantages" in inputs and torch.abs(inputs["advantages"]).max() < 1e-6:
            # Skip expensive forward/backward passes - no learning signal
            return torch.tensor(0.0, device=inputs["advantages"].device)

        # Track effective training steps (non-zero advantages)
        self.effective_steps += 1

        # Normal training step for non-zero advantages
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        model.train()

        # Note that the output of the policy
        # is a list containing pointers to tensors of size  (B, T, BL)
        # although if you sum the B dim you get the group size G, these can
        # in general not be stacked because the T
        # can vary between different batches within the group.
        # Therefore, we process it as a list of batches for as long as necessary.
        policy_outputs: list[dict[str, torch.Tensor | tuple[torch.Tensor]]] = inputs[
            "policy_outputs"
        ]

        # ensure the group batches add up to the whole group size (ie sum_i B_i = G)
        group_batch_sizes = [
            policy_output["sampling_masks"].size(0) for policy_output in policy_outputs
        ]
        assert sum(group_batch_sizes) == inputs["advantages"].size(0)

        # Check if ES (Expert Steering) is enabled and compute mixture distribution weights
        has_es = (
            self.args.es_thresholds is not None and len(self.args.es_thresholds) > 0
        )
        if has_es:
            num_es = len(self.args.es_thresholds)
            total_group_size = inputs["advantages"].size(0)
            num_regular = total_group_size - num_es
            device = inputs["advantages"].device
            # Mixture weights: (G/(G+E)) * pi_theta + (1/(G+E)) * dirac
            # Note that we assume a sample can only come from one particular ES dirac
            # (hence why the second factor is not E/(G+E))
            log_weight_theta = torch.log(
                torch.tensor(num_regular / total_group_size, device=device)
            )
            log_weight_dirac = torch.log(
                torch.tensor(1.0 / total_group_size, device=device)
            )

        # Accumulate the loss over the batches in the group
        batch_index_start = 0
        loss_acummulator = 0
        entropy_accumulator = []
        for batch_idx, batch_policy_output in enumerate(policy_outputs):
            batch_sampling_masks = batch_policy_output[
                "sampling_masks"
            ]  # will need these later

            batch_index_end = batch_index_start + batch_sampling_masks.size(0)

            # Detect if this is an ES batch (last batch in the group)
            is_es_batch = has_es and batch_idx == len(policy_outputs) - 1

            B, T, _ = batch_sampling_masks.shape

            time_step_loss_accumulator = torch.zeros(
                B, device=batch_sampling_masks.device
            )  # (B,)
            timestep_bs = (
                self.args.timestep_batch_size
                if self.args.timestep_batch_size is not None
                else T
            )
            for time_step_idx in range(0, T, timestep_bs):
                # Get this batch's data
                time_step_batch_sampling_masks = batch_sampling_masks[
                    :, time_step_idx : time_step_idx + timestep_bs, :
                ]
                time_step_batch_samples = batch_policy_output["samples"][
                    :, time_step_idx : time_step_idx + timestep_bs, :
                ]

                ### Prepare time-batched inputs
                time_step_batch_policy_inputs = []
                for ptdi in batch_policy_output["policy_inputs"]:
                    if isinstance(ptdi, torch.Tensor):
                        # Find time dim and slice only at that dim
                        time_dim = 1
                        assert ptdi.size(time_dim) == T, (
                            f"ptdi of shape {ptdi.shape=} did not match {T=} "
                            f"at expected {time_dim=}"
                        )
                        slices = [slice(None)] * ptdi.ndim  # by default get everything
                        slices[time_dim] = slice(
                            time_step_idx, time_step_idx + timestep_bs
                        )  # at time_dim get only what belongs to this time-batch

                        time_step_batch_policy_inputs.append(ptdi[tuple(slices)])
                    else:
                        # Non-tensors just get propagated
                        time_step_batch_policy_inputs.append(ptdi)

                logps_timestep = self._get_per_timestep_logps_block(
                    model=model,
                    samples=time_step_batch_samples,
                    sampling_masks=time_step_batch_sampling_masks,
                    policy_inputs=time_step_batch_policy_inputs,
                    sampling_mode=self.args.sampling_mode,
                    return_entropy=True,
                )  # (B, timestep_bs), entropy (scalar)

                if isinstance(logps_timestep, tuple):
                    logps_timestep, entropy = logps_timestep
                    entropy_accumulator.append(entropy)
                else:
                    # Backward compatibility if return_entropy=False
                    pass

                # For ES batches, adjust NEW log probabilities with mixture distribution
                if is_es_batch:
                    logps_timestep = torch.logaddexp(
                        log_weight_theta + logps_timestep, log_weight_dirac
                    )

                # Get old log probabilities
                old_logps_slice = batch_policy_output["old_per_timestep_logps"][
                    :, time_step_idx : time_step_idx + timestep_bs
                ].detach()

                # For ES batches, adjust OLD log probabilities with mixture distribution
                if is_es_batch:
                    old_logps_slice = torch.logaddexp(
                        log_weight_theta + old_logps_slice, log_weight_dirac
                    )

                coeff_1 = torch.exp(
                    logps_timestep - old_logps_slice
                )  # (B, timestep_bs)
                coeff_2 = torch.clamp(
                    coeff_1, 1 - self.args.epsilon, 1 + self.args.epsilon
                )

                # Get the advantages corresponding to this batch.
                # Note that the "advantages" are flat of shape (G,),
                # so we do some indexing to keep track of where the
                # current batch is.
                batch_advantages = (
                    inputs["advantages"][batch_index_start:batch_index_end]
                    .detach()
                    .view((-1,) + (1,) * (coeff_1.ndim - 1))
                )  # (B, 1)

                per_timestep_loss1 = coeff_1 * batch_advantages
                per_timestep_loss2 = coeff_2 * batch_advantages
                per_timestep_loss = torch.min(per_timestep_loss1, per_timestep_loss2)

                # Only include the loss for the active timesteps
                per_timestep_loss *= time_step_batch_sampling_masks.any(dim=-1).to(
                    per_timestep_loss.dtype
                )
                time_step_loss_accumulator += per_timestep_loss.sum(dim=-1)

                del (
                    logps_timestep,
                    coeff_1,
                    coeff_2,
                    per_timestep_loss1,
                    per_timestep_loss2,
                    per_timestep_loss,
                )
                torch.cuda.empty_cache()

            num_active_steps = batch_sampling_masks.any(dim=-1).sum(dim=-1)  # (B,)
            assert (num_active_steps > 0).all(), (
                "At least one batch element was active for < 1 steps?"
            )
            batch_loss = time_step_loss_accumulator / num_active_steps

            # The accumulated loss is updated per the sum of -batch_loss
            # (we later turn the sum into a mean by dividing by the group size)
            loss_acummulator -= batch_loss.sum()

            # Next batch starts where the current one left off
            batch_index_start = batch_index_end

        assert self.beta == 0.0, (
            f"TODO non-zero {self.beta=} not supported at this time"
        )

        # final loss is average over the group
        loss = loss_acummulator / sum(group_batch_sizes)

        # Log entropy if collected
        if entropy_accumulator:
            mean_entropy = torch.stack(entropy_accumulator).mean()
            self._metrics["train"]["entropy"].append(
                self.accelerator.gather_for_metrics(mean_entropy).mean().item()
            )

        return loss

    def _get_per_timestep_logps_block(
        self,
        model,
        samples,
        sampling_masks,
        policy_inputs,
        sampling_mode="bernoulli",
        return_entropy=False,
    ):
        """Compute log-probabilities for sampled actions under the policy.

        :param model: policy model
        :param samples: sampled actions
        :param sampling_masks: mask indicating valid positions
        :param policy_inputs: inputs to policy model
        :param sampling_mode: sampling mode ("bernoulli", "dpls")
        :param return_entropy: whether to compute and return entropy
        :return: (log_probs, entropy) or just log_probs if return_entropy=False
        """
        with torch.amp.autocast("cuda", enabled=self.args.fp16):
            logits = model(*policy_inputs)  # (B, T, BL)

        # Calculate corresponding log-likelihoods under the model
        if sampling_mode == "dpls":
            lls = dpls_batch_loglik(
                samples=samples,
                utilities=logits,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif sampling_mode in ["bernoulli", "bernoulli-argmax"]:
            lls = bernoulli_batch_loglik(
                samples,
                logits,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unexpected {sampling_mode=}")

        # Compute entropy if requested
        entropy = None
        if return_entropy:
            if sampling_mode in ["bernoulli", "bernoulli-argmax"]:
                # For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
                probs_clamped = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)
                entropy = -(
                    probs_clamped * torch.log(probs_clamped)
                    + (1 - probs_clamped) * torch.log(1 - probs_clamped)
                )  # (B, timestep_bs, BL)
                # Average over active positions only
                active_mask = sampling_masks.float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()
            elif sampling_mode == "dpls":
                masked_logits = logits.masked_fill(~sampling_masks, float("-inf"))
                probs = torch.softmax(masked_logits, dim=-1)
                probs_clamped = probs.clamp(1e-8, 1.0)
                entropy = -(probs * torch.log(probs_clamped)).sum(dim=-1)
                active_mask = sampling_masks.any(dim=-1).float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()

        del logits
        torch.cuda.empty_cache()

        if return_entropy:
            return lls, entropy
        return lls

    def _compute_mask_loglikelihood(
        self,
        samples: torch.Tensor,
        sampling_inputs: torch.Tensor,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.args.sampling_mode in ["bernoulli", "bernoulli-argmax"]:
            return bernoulli_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif self.args.sampling_mode == "dpls":
            return dpls_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {self.args.sampling_mode}")

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                # Generate new completions
                inputs = self._generate_and_score_completions(inputs)
                # Store for reuse in next num_iterations-1 steps
                self._buffered_inputs = inputs
            else:
                # Reuse buffered completions
                inputs = self._buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        # Remove assertion - we now support mixed dataset types in a batch
        # assert len(set([x["dataset_type"] for x in inputs])) == 1

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # Need to add the gen_prefix to the prompt for KodCode (per-sample check)
        for i, example in enumerate(inputs):
            if example["dataset_type"] == "kodcode":
                prompts_text[i] = prompts_text[i] + example["gen_prefix"]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = HFTrainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        temperature = self.args.temperature or 0.0
        budget_choices = self.args.train_budget_sampling or [self.args.target_budget]
        if self.args.enable_budget_conditioning:
            target_budgets_for_policy = sample_group_target_budgets(
                expanded_batch_size=prompt_ids.size(0),
                num_generations=self.num_generations,
                budget_choices=budget_choices,
                device=device,
            )
        else:
            target_budgets_for_policy = torch.full(
                (prompt_ids.size(0),),
                int(self.args.target_budget),
                device=device,
                dtype=torch.long,
            )
        rollout_compute_styles = assign_rollout_compute_styles(
            expanded_batch_size=prompt_ids.size(0),
            num_generations=self.num_generations,
            styles=self.args.train_rollout_compute_styles,
        )

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            num_steps_all = []
            remask_counts_all = []
            target_budgets_all = []
            metadata_all = []
            force_es_thresholds = None
            if self.args.es_thresholds:
                # TODO: For now we are hardcoding BL=32 for ES samples.
                force_es_thresholds = torch.tensor(
                    self.args.es_thresholds,
                    dtype=unwrapped_model.dtype,
                    device=unwrapped_model.device,
                ).unsqueeze(-1)
            if self.args.remasking == "policy":
                policy_outputs_all = []
                still_masked_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_prompt_mask = prompt_mask[i:end_idx]
                    batch_target_budget = target_budgets_for_policy[i:end_idx]
                    batch_rollout_styles = rollout_compute_styles[i:end_idx]
                    generation_target_budget = (
                        batch_target_budget
                        if self.args.enable_budget_conditioning
                        else None
                    )
                    effective_hard_stop = (
                        self.args.hard_stop_at_target_budget
                        and generation_target_budget is not None
                    )

                    result = generate_unified(
                        model=self.dllm,
                        prompt=batch_prompt_ids,
                        remasking="policy",
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        sampling_mode=self.args.sampling_mode,
                        dpls_stop_logit=self.args.dpls_stop_logit,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        model_type=self.args.model_type,
                        attention_mask=batch_prompt_mask,
                        tokenizer=self.processing_class
                        if (
                            self.args.enable_verifier
                            or self.args.enable_posthoc_continuation
                        )
                        else None,
                        enable_verifier=self.args.enable_verifier,
                        verifier_type=self.args.verifier_type,
                        verifier_schedule=self.args.verifier_schedule,
                        verifier_every_k_steps=self.args.verifier_every_k_steps,
                        verifier_threshold=self.args.verifier_threshold,
                        enable_remasking=self.args.enable_remasking,
                        remask_strategy=self.args.remask_strategy,
                        max_remasks_per_sample=self.args.max_remasks_per_sample,
                        remask_cooldown_steps=self.args.remask_cooldown_steps,
                        min_steps_before_remask=self.args.min_steps_before_remask,
                        remask_answer_span_mode=self.args.remask_answer_span_mode,
                        target_budget=generation_target_budget,
                        hard_stop_at_target_budget=effective_hard_stop,
                        hard_generation_budget=self.args.hard_generation_budget,
                        policy_extra_feature_names=get_policy_extra_feature_names(
                            self.args
                        ),
                        log_step_traces=self.args.log_step_traces,
                        rollout_compute_style=batch_rollout_styles,
                        patient_unmask_logit_bias=self.args.patient_unmask_logit_bias,
                        patient_policy_temperature=self.args.patient_policy_temperature,
                        patient_max_unmask_fraction=(
                            self.args.patient_max_unmask_fraction
                        ),
                        patient_min_steps_frac=self.args.patient_min_steps_frac,
                        patient_global_slowdown=self.args.patient_global_slowdown,
                        patient_delay_low_confidence=(
                            self.args.patient_delay_low_confidence
                        ),
                        patient_low_confidence_quantile=(
                            self.args.patient_low_confidence_quantile
                        ),
                        patient_delay_final_window=(
                            self.args.patient_delay_final_window
                        ),
                        patient_final_window_tokens=(
                            self.args.patient_final_window_tokens
                        ),
                        patient_final_window_min_steps_frac=(
                            self.args.patient_final_window_min_steps_frac
                        ),
                        enable_deliberation_probe=self.args.enable_deliberation_probe,
                        deliberation_probe_frac=self.args.deliberation_probe_frac,
                        enable_posthoc_continuation=(
                            self.args.enable_posthoc_continuation
                        ),
                        continuation_steps=list(self.args.continuation_steps),
                        continuation_trigger=self.args.continuation_trigger,
                        continuation_remask_mode=self.args.continuation_remask_mode,
                        continuation_final_window_tokens=(
                            self.args.continuation_final_window_tokens
                        ),
                        continuation_low_confidence_quantile=(
                            self.args.continuation_low_confidence_quantile
                        ),
                        continuation_min_remaining_budget=(
                            self.args.continuation_min_remaining_budget
                        ),
                        continuation_use_oracle_for_analysis=(
                            self.args.continuation_use_oracle_for_analysis
                        ),
                        continuation_force_for_analysis=(
                            self.args.continuation_force_for_analysis
                        ),
                        continuation_select_best_k_oracle=(
                            self.args.continuation_select_best_k_oracle
                        ),
                        continuation_confidence_threshold=(
                            self.args.continuation_confidence_threshold
                        ),
                        continuation_verifier_threshold=(
                            self.args.continuation_verifier_threshold
                        ),
                    )

                    # Extract values from NamedTuple
                    batch_prompt_completion_ids = result.sequences
                    batch_sampling_inputs = result.sampling_inputs
                    batch_samples = result.samples
                    batch_sampling_masks = result.sampling_masks
                    num_steps = result.steps_taken
                    batch_policy_inputs = result.policy_inputs
                    still_masked = result.still_masked

                    # Compute log-likelihood based on sampling mode
                    mask_ll = self._compute_mask_loglikelihood(
                        samples=batch_samples,
                        sampling_inputs=batch_sampling_inputs,
                        sampling_masks=batch_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": batch_samples,
                            "sampling_masks": batch_sampling_masks,
                            "old_per_timestep_logps": mask_ll,
                            "prompt_length": batch_prompt_ids.shape[1],
                            "sampling_inputs": batch_sampling_inputs,
                            "policy_inputs": batch_policy_inputs,
                        }
                    )
                    num_steps_all.append(num_steps)
                    still_masked_all.append(still_masked)
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)
                    batch_metadata = result.metadata or []
                    for local_idx, item in enumerate(batch_metadata):
                        global_idx = i + local_idx
                        item["target_budget"] = int(
                            batch_target_budget[local_idx].item()
                        )
                        item["budget_error"] = item.get("nfe_used", 0) - item[
                            "target_budget"
                        ]
                        item["group_id"] = global_idx // self.num_generations
                        item["rollout_index"] = global_idx % self.num_generations
                        item["rollout_compute_style"] = rollout_compute_styles[
                            global_idx
                        ]
                        item["num_generations"] = self.num_generations
                    metadata_all.extend(batch_metadata)
                    remask_counts_all.append(
                        torch.tensor(
                            [
                                item.get("remask_count", 0)
                                for item in (result.metadata or [])
                            ],
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    target_budgets_all.append(batch_target_budget)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                if force_es_thresholds is not None:
                    es_prompt_ids = (
                        prompt_ids[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_ids.ndim - 1),
                        )
                        .contiguous()
                    )
                    es_prompt_mask = (
                        prompt_mask[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_mask.ndim - 1),
                        )
                        .contiguous()
                    )
                    es_target_budget = target_budgets_for_policy[0].repeat(
                        es_prompt_ids.size(0)
                    )
                    es_generation_target_budget = (
                        es_target_budget
                        if self.args.enable_budget_conditioning
                        else None
                    )
                    es_effective_hard_stop = (
                        self.args.hard_stop_at_target_budget
                        and es_generation_target_budget is not None
                    )
                    result = generate_unified(
                        model=self.dllm,
                        prompt=es_prompt_ids,
                        remasking="fastdllm",
                        thres=force_es_thresholds,
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=32,  # TODO: in the future we may not want to hardcode this
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        temperature_policy=1.0,
                        model_type=self.args.model_type,
                        attention_mask=es_prompt_mask,
                        tokenizer=self.processing_class
                        if (
                            self.args.enable_verifier
                            or self.args.enable_posthoc_continuation
                        )
                        else None,
                        enable_verifier=self.args.enable_verifier,
                        verifier_type=self.args.verifier_type,
                        verifier_schedule=self.args.verifier_schedule,
                        verifier_every_k_steps=self.args.verifier_every_k_steps,
                        verifier_threshold=self.args.verifier_threshold,
                        enable_remasking=self.args.enable_remasking,
                        remask_strategy=self.args.remask_strategy,
                        max_remasks_per_sample=self.args.max_remasks_per_sample,
                        remask_cooldown_steps=self.args.remask_cooldown_steps,
                        min_steps_before_remask=self.args.min_steps_before_remask,
                        remask_answer_span_mode=self.args.remask_answer_span_mode,
                        target_budget=es_generation_target_budget,
                        hard_stop_at_target_budget=es_effective_hard_stop,
                        hard_generation_budget=self.args.hard_generation_budget,
                        policy_extra_feature_names=get_policy_extra_feature_names(
                            self.args
                        ),
                        log_step_traces=self.args.log_step_traces,
                        rollout_compute_style=["normal"] * es_prompt_ids.size(0),
                        patient_global_slowdown=self.args.patient_global_slowdown,
                        patient_delay_low_confidence=(
                            self.args.patient_delay_low_confidence
                        ),
                        patient_low_confidence_quantile=(
                            self.args.patient_low_confidence_quantile
                        ),
                        patient_delay_final_window=(
                            self.args.patient_delay_final_window
                        ),
                        patient_final_window_tokens=(
                            self.args.patient_final_window_tokens
                        ),
                        patient_final_window_min_steps_frac=(
                            self.args.patient_final_window_min_steps_frac
                        ),
                        enable_deliberation_probe=self.args.enable_deliberation_probe,
                        deliberation_probe_frac=self.args.deliberation_probe_frac,
                        enable_posthoc_continuation=(
                            self.args.enable_posthoc_continuation
                        ),
                        continuation_steps=list(self.args.continuation_steps),
                        continuation_trigger=self.args.continuation_trigger,
                        continuation_remask_mode=self.args.continuation_remask_mode,
                        continuation_final_window_tokens=(
                            self.args.continuation_final_window_tokens
                        ),
                        continuation_low_confidence_quantile=(
                            self.args.continuation_low_confidence_quantile
                        ),
                        continuation_min_remaining_budget=(
                            self.args.continuation_min_remaining_budget
                        ),
                        continuation_use_oracle_for_analysis=(
                            self.args.continuation_use_oracle_for_analysis
                        ),
                        continuation_force_for_analysis=(
                            self.args.continuation_force_for_analysis
                        ),
                        continuation_select_best_k_oracle=(
                            self.args.continuation_select_best_k_oracle
                        ),
                        continuation_confidence_threshold=(
                            self.args.continuation_confidence_threshold
                        ),
                        continuation_verifier_threshold=(
                            self.args.continuation_verifier_threshold
                        ),
                    )

                    # Extract values from NamedTuple
                    es_prompt_completion_ids = result.sequences
                    es_sampling_inputs = result.sampling_inputs
                    es_samples = result.samples
                    es_sampling_masks = result.sampling_masks
                    es_num_steps = result.steps_taken
                    es_policy_inputs = result.policy_inputs
                    # still_masked ignored for ES

                    # Compute log-likelihood based on sampling mode
                    es_mask_ll = self._compute_mask_loglikelihood(
                        samples=es_samples,
                        sampling_inputs=es_sampling_inputs,
                        sampling_masks=es_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": es_samples,
                            "sampling_masks": es_sampling_masks,
                            "old_per_timestep_logps": es_mask_ll,
                            "prompt_length": es_prompt_ids.shape[1],
                            "sampling_inputs": es_sampling_inputs,
                            "policy_inputs": es_policy_inputs,
                        }
                    )
                    num_steps_all.append(es_num_steps)
                    prompt_completion_ids_all.append(es_prompt_completion_ids)
                    es_metadata = result.metadata or []
                    for local_idx, item in enumerate(es_metadata):
                        item["target_budget"] = int(es_target_budget[local_idx].item())
                        item["budget_error"] = item.get("nfe_used", 0) - item[
                            "target_budget"
                        ]
                        item["group_id"] = 0
                        item["rollout_index"] = self.num_generations + local_idx
                        item["rollout_compute_style"] = "expert_steering"
                        item["num_generations"] = self.num_generations
                        item["expert_steering"] = True
                    metadata_all.extend(es_metadata)
                    remask_counts_all.append(
                        torch.tensor(
                            [
                                item.get("remask_count", 0)
                                for item in (result.metadata or [])
                            ],
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    target_budgets_all.append(es_target_budget)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
                num_steps = torch.cat(num_steps_all, dim=0)
                remask_counts = torch.cat(remask_counts_all, dim=0)
                target_budgets = torch.cat(target_budgets_all, dim=0)
                still_masked = torch.cat(still_masked_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        if self.args.es_thresholds:
            # Add copies for ES, if needed
            # TODO: assuming we can reuse inputs[0],
            # should be safe since all inputs should be the same on the gpu anyway
            num_es_samples = len(self.args.es_thresholds)
            inputs.extend([inputs[0]] * num_es_samples)
            prompts.extend([inputs[0]["prompt"]] * num_es_samples)
        assert len(completion_ids) == len(prompts) == len(inputs), (
            f"{len(completion_ids)=} ?!= {len(prompts)=} ?!= {len(inputs)=}"
        )

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        effective_reward_budget_mode = _effective_reward_budget_mode(self.args)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            if reward_func_name in [
                "lm_eval_flex_mult_reward",
                "lm_eval_flex_add_reward",
                "xml_mult_reward",
                "xml_add_reward",
                "math_correctness_mult_reward",
                "mixed_correctness_mult_reward_func",
                "mixed_correctness_add_reward_func",
                "mixed_correctness_reward_func",
                "mixed_additive_proposal_reward_func",
                "mixed_multiplicative_budget_reward_func",
                "kodcode_correctness_mult_reward",
            ]:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    n_steps=num_steps,
                    L=gen_length,
                    alpha=self.args.alpha_compute_reward,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    target_budget=target_budgets,
                    reward_lambda_compute=self.args.reward_lambda_compute,
                    reward_normalize_compute=self.args.reward_normalize_compute,
                    reward_gsm_extractor=self.args.reward_gsm_extractor,
                    reward_beta=self.args.reward_beta,
                    reward_mu=self.args.reward_mu,
                    reward_rho=self.args.reward_rho,
                    reward_quality_mode=self.args.reward_quality_mode,
                    answer_span_partial_credit=self.args.answer_span_partial_credit,
                    malformed_answer_factor=self.args.malformed_answer_factor,
                    source_answer_marker_credit=self.args.source_answer_marker_credit,
                    source_final_line_credit=self.args.source_final_line_credit,
                    source_final_window_credit=self.args.source_final_window_credit,
                    source_incomplete_answer_tag_credit=(
                        self.args.source_incomplete_answer_tag_credit
                    ),
                    source_other_answer_span_credit=(
                        self.args.source_other_answer_span_credit
                    ),
                    enable_repetition_penalty=self.args.enable_repetition_penalty,
                    repetition_min_tokens=self.args.repetition_min_tokens,
                    repetition_number_run_threshold=(
                        self.args.repetition_number_run_threshold
                    ),
                    repetition_token_run_threshold=(
                        self.args.repetition_token_run_threshold
                    ),
                    repetition_bigram_run_threshold=(
                        self.args.repetition_bigram_run_threshold
                    ),
                    repetition_answer_span_number_threshold=(
                        self.args.repetition_answer_span_number_threshold
                    ),
                    repetition_unique_ratio_threshold=(
                        self.args.repetition_unique_ratio_threshold
                    ),
                    repetition_max_freq_threshold=(
                        self.args.repetition_max_freq_threshold
                    ),
                    repetition_penalty_factor=self.args.repetition_penalty_factor,
                    zero_reward_if_repeated_answer_span=(
                        self.args.zero_reward_if_repeated_answer_span
                    ),
                    max_reward_if_malformed=self.args.max_reward_if_malformed,
                    max_reward_if_repetitive=self.args.max_reward_if_repetitive,
                    reward_budget_mode=effective_reward_budget_mode,
                    reward_type=self.args.reward_type,
                    threshold_reward_hard_zero=self.args.threshold_reward_hard_zero,
                    enable_negative_junk_penalty=(
                        self.args.enable_negative_junk_penalty
                    ),
                    repeated_junk_negative_reward=(
                        self.args.repeated_junk_negative_reward
                    ),
                    malformed_junk_negative_reward=(
                        self.args.malformed_junk_negative_reward
                    ),
                    incomplete_answer_tag_negative_reward=(
                        self.args.incomplete_answer_tag_negative_reward
                    ),
                    min_reward_floor=self.args.min_reward_floor,
                    format_ok=[item.get("format_ok") for item in metadata_all],
                    early_generated_text=[
                        item.get("early_generated_text") for item in metadata_all
                    ],
                    rollout_compute_style=[
                        item.get("rollout_compute_style", "normal")
                        for item in metadata_all
                    ],
                    num_generations=self.num_generations,
                    enable_deliberation_probe=self.args.enable_deliberation_probe,
                    deliberation_bonus_weight=self.args.deliberation_bonus_weight,
                    patient_win_bonus=self.args.patient_win_bonus,
                    patient_win_margin=self.args.patient_win_margin,
                    remask_count=remask_counts,
                    **reward_kwargs,
                )
            elif reward_func_name == "lm_eval_harness_flexible_match_reward_func":
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    **reward_kwargs,
                )
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )

            assert len(output_reward_func) == len(prompts) == len(completion_ids), (
                f"{len(output_reward_func)=} != {len(prompts)=} = {len(completion_ids)=}"
            )
            # Convert None values to NaN
            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            assert len(output_reward_func) == len(prompts), (
                f"{len(output_reward_func)=} != {len(prompts)=}"
            )

            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        local_rewards_per_func = rewards_per_func.detach().clone()
        local_rewards = (
            local_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        mode = "eval" if self.control.should_evaluate else "train"
        self._write_generation_records(
            mode=mode,
            inputs=inputs,
            prompts_text=prompts_text,
            completions_text=completions_text,
            metadata=metadata_all,
            rewards=local_rewards,
            rewards_per_func=local_rewards_per_func,
        )

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        group_size = self.num_generations + len(self.args.es_thresholds or [])
        grouped_rewards = rewards.view(-1, group_size)
        mean_grouped_rewards_raw = grouped_rewards.mean(dim=1)
        std_grouped_rewards_raw = grouped_rewards.std(dim=1, unbiased=False)
        zero_reward_groups = (grouped_rewards.abs().amax(dim=1) <= 1e-8)
        mixed_reward_groups = (grouped_rewards.amin(dim=1) <= 1e-8) & (
            grouped_rewards.amax(dim=1) > 1e-8
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards_raw.repeat_interleave(
            group_size,
            dim=0,
        )
        std_grouped_rewards = std_grouped_rewards_raw.repeat_interleave(
            group_size,
            dim=0,
        )
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation (policy only for metrics)
        zero_std_count = (std_grouped_rewards_raw < 1e-6).sum().item()
        total_prompts = std_grouped_rewards_raw.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        # Slice out this process's advantages. TRL can pass already-expanded
        # GRPO samples here, so use the actual local sample count instead of
        # assuming it equals per_device_train_batch_size.
        items_per_process = len(prompts)
        process_slice = slice(
            self.accelerator.process_index * items_per_process,
            (self.accelerator.process_index + 1) * items_per_process,
        )
        advantages = advantages[process_slice]

        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float()

        # For rewards and other metrics (eg completion length) inferred from the prompt_completion_ids,
        # we need to slice out ES samples (if any) so as to not pollute the logs
        num_es = len(self.args.es_thresholds) if self.args.es_thresholds else 0
        if num_es > 0:
            num_processes = self.accelerator.num_processes
            policy_samples_per_process = group_size - num_es
            post_gathering_policy_only_index = torch.ones(
                completion_length.shape[0],
                dtype=torch.bool,
                device=completion_length.device,
            )
            for proc_idx in range(num_processes):
                start_idx = proc_idx * group_size
                es_start = start_idx + policy_samples_per_process
                es_end = start_idx + group_size
                post_gathering_policy_only_index[es_start:es_end] = False
        else:
            post_gathering_policy_only_index = slice(None)

        completion_length = (
            completion_length[post_gathering_policy_only_index].mean().item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
        self._metrics[mode]["effective_steps"].append(self.effective_steps)

        still_masked_metric = gather(still_masked).float()
        self._metrics[mode]["still_masked_sample_rate"].append(
            still_masked_metric.mean().item()
        )
        policy_metadata_for_metrics = metadata_all[: still_masked.numel()]
        masked_token_counts = torch.tensor(
            [
                item.get("masked_token_count", 0)
                for item in policy_metadata_for_metrics
            ],
            device=device,
            dtype=torch.float32,
        )
        masked_token_fractions = torch.tensor(
            [
                item.get("masked_token_fraction", 0.0)
                for item in policy_metadata_for_metrics
            ],
            device=device,
            dtype=torch.float32,
        )
        masked_token_counts = self.accelerator.gather_for_metrics(masked_token_counts)
        masked_token_fractions = self.accelerator.gather_for_metrics(
            masked_token_fractions
        )
        self._metrics[mode]["masked_token_count_mean"].append(
            masked_token_counts.mean().item()
        )
        self._metrics[mode]["masked_token_fraction_mean"].append(
            masked_token_fractions.mean().item()
        )

        # Metrics: Calculate mean reward per function, but only for samples where the function was applied
        # and the sample actually came from the policy (not ES)
        rewards_per_func_policy = rewards_per_func[post_gathering_policy_only_index]
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func_policy[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        rewards_policy = rewards[post_gathering_policy_only_index]
        self._metrics[mode]["reward"].append(rewards_policy.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards_raw.mean().item())
        self._metrics[mode]["reward_max"].append(rewards_policy.max().item())
        raw_task_score_values = []
        budget_multiplier_values = []
        for idx, example in enumerate(inputs):
            item_metadata = metadata_all[idx] if idx < len(metadata_all) else {}
            generated_text = (
                completions_text[idx]
                if idx < len(completions_text)
                else item_metadata.get("generated_text_final")
            )
            if example.get("dataset_type", self.args.dataset) == "gsm8k":
                fields = _gsm_reward_parse_fields(
                    generated_text,
                    example.get("answer"),
                    self.args,
                    item_metadata,
                )
                raw_task_score_values.append(float(fields["raw_task_score"]))
                budget_multiplier_values.append(float(fields["budget_multiplier"]))
            else:
                raw_task_score_values.append(float("nan"))
                budget_multiplier_values.append(float("nan"))
        raw_task_scores_metric = self.accelerator.gather_for_metrics(
            torch.tensor(raw_task_score_values, device=device, dtype=torch.float32)
        )[post_gathering_policy_only_index]
        budget_multipliers_metric = self.accelerator.gather_for_metrics(
            torch.tensor(budget_multiplier_values, device=device, dtype=torch.float32)
        )[post_gathering_policy_only_index]
        finite_raw_task_scores = raw_task_scores_metric[
            torch.isfinite(raw_task_scores_metric)
        ]
        if finite_raw_task_scores.numel() > 0:
            self._metrics[mode]["raw_task_score_mean"].append(
                finite_raw_task_scores.mean().item()
            )
            self._metrics[mode]["raw_task_score_max"].append(
                finite_raw_task_scores.max().item()
            )
        finite_budget_multipliers = budget_multipliers_metric[
            torch.isfinite(budget_multipliers_metric)
        ]
        if finite_budget_multipliers.numel() > 0:
            self._metrics[mode]["budget_multiplier_mean"].append(
                finite_budget_multipliers.mean().item()
            )
            self._metrics[mode]["budget_multiplier_min"].append(
                finite_budget_multipliers.min().item()
            )
            self._metrics[mode]["budget_factor_mean"].append(
                finite_budget_multipliers.mean().item()
            )
        self._metrics[mode]["positive_reward_rate"].append(
            (rewards_policy > 1e-8).float().mean().item()
        )
        self._metrics[mode]["group_reward_mean"].append(
            mean_grouped_rewards_raw.mean().item()
        )
        self._metrics[mode]["group_reward_std"].append(
            std_grouped_rewards_raw.mean().item()
        )
        self._metrics[mode]["zero_reward_group_rate"].append(
            zero_reward_groups.float().mean().item()
        )
        self._metrics[mode]["mixed_reward_group_rate"].append(
            mixed_reward_groups.float().mean().item()
        )
        target_budgets_metric = self.accelerator.gather_for_metrics(
            target_budgets
        ).float()
        num_steps_for_budget = self.accelerator.gather_for_metrics(num_steps).float()
        target_budgets_policy = target_budgets_metric[post_gathering_policy_only_index]
        num_steps_budget_policy = num_steps_for_budget[post_gathering_policy_only_index]
        self._metrics[mode]["target_budget"].append(
            target_budgets_policy.mean().item()
        )
        budget_error = num_steps_budget_policy - target_budgets_policy
        self._metrics[mode]["budget_error_mean"].append(budget_error.mean().item())
        self._metrics[mode]["budget_error_abs_mean"].append(
            budget_error.abs().mean().item()
        )
        for budget_value in sorted(set(int(v.item()) for v in target_budgets_policy)):
            budget_mask = target_budgets_policy == budget_value
            if budget_mask.any():
                self._metrics[mode][
                    f"avg_nfe_by_target_budget/{budget_value}"
                ].append(num_steps_budget_policy[budget_mask].mean().item())
        self._metrics[mode]["reward_type/multiplicative_budget"].append(
            float(self.args.reward_type == "multiplicative_budget")
        )
        self._metrics[mode]["reward_type/additive_proposal"].append(
            float(self.args.reward_type == "additive_proposal")
        )
        self._metrics[mode]["reward_type/task_only"].append(
            float(self.args.reward_type == "task_only")
        )
        self._metrics[mode]["budget_is_hard_cap"].append(
            float(
                self.args.enable_budget_conditioning
                and self.args.hard_stop_at_target_budget
            )
        )

        # Log ES advantages if present
        if num_es > 0:
            # Gather all advantages across processes
            advantages_gathered = gather(advantages)
            # Extract ES advantages (inverse of policy-only index)
            advantages_es = advantages_gathered[~post_gathering_policy_only_index]
            self._metrics[mode]["es_advantage_mean"].append(advantages_es.mean().item())

        if (
            self.args.save_best_checkpoint
            and mode == "train"
            and self.accelerator.is_main_process
            and self.state.global_step % self.num_iterations == 0
        ):
            self.train_reward_queue.append(rewards_policy.mean().item())
            if np.mean(self.train_reward_queue) > self.train_reward_best:
                self.train_reward_best = np.mean(self.train_reward_queue)
                self.train_reward_best_step = self.state.global_step

                _output_dir = os.path.join(self.args.output_dir, "checkpoint-best")
                unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
                unwrapped_model.save_pretrained(_output_dir)
                self.state.save_to_json(os.path.join(_output_dir, "trainer_state.json"))
                with open(
                    os.path.join(_output_dir, "best_train_reward.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "best_train_reward": self.train_reward_best,
                            "best_train_reward_step": self.train_reward_best_step,
                        },
                        f,
                    )

                print(
                    f"Saved checkpoint-best at step {self.train_reward_best_step} with train reward {self.train_reward_best}"
                )
                if self.s3_callback is not None:
                    # use the callback to push the checkpoint to s3
                    print(f"Uploading checkpoint-best to s3: {self.args.output_dir}")
                    self.s3_callback.on_save(
                        self.args, self.state, self.control, best=True
                    )

        # Log metrics to detect the collapse to 0 policy
        avg_us_all = []
        max_us_all = []
        non_zero_active_us_all = []
        non_zero_bs_timesteps_all = []
        for i in range(
            # Drop last batch, corresponding to ES samples, if present
            len(policy_outputs_all) - (1 if self.args.es_thresholds else 0)
        ):
            sampling_inputs = policy_outputs_all[i][
                "sampling_inputs"
            ]  # Contains logits for both Bernoulli and DPLS
            samples = policy_outputs_all[i][
                "samples"
            ]  # One-hot bernoulli outcomes for Bernoulli, ordered indices for PL
            ms = policy_outputs_all[i]["sampling_masks"]

            # Convert to probabilities for consistent logging across sampling modes
            if self.args.sampling_mode == "dpls":
                # For DPLS, sampling_inputs contains logits, convert to probabilities
                # Do not normalize over unmasked tokens
                sampling_inputs = torch.where(
                    ms.any(dim=-1).unsqueeze(-1),
                    sampling_inputs.masked_fill(~ms, float("-inf")),
                    torch.zeros_like(sampling_inputs),
                )
                us = torch.softmax(sampling_inputs, dim=-1, dtype=torch.float32)
                # Similarly samples need to be converted to one-hot
                # (do not care about their ordering for logging)
                # Since samples vector may contain padding (-1), we clamp to valid indices
                # but then dynamically set the value to True/False - making the scatter
                # a no-op at the padded indices
                bs = (
                    torch.zeros_like(us, dtype=torch.int)
                    .scatter_add(-1, samples.clamp(min=0), (samples >= 0).int())
                    .bool()
                )
            else:
                # For Bernoulli, sampling_inputs contains logits, convert to probabilities with sigmoid
                us = torch.sigmoid(sampling_inputs)
                # And samples contains the sampled unmasking indices (one-hot)
                bs = samples

            # For average unmask probability as well as for proportion of non-zero unmask probability,
            # we average both over time and the block dimension
            avg_us = (us * ms).sum(dim=(-1, -2)) / ms.sum(dim=(-1, -2))
            eps = 0.001
            non_zero_active_us = ((us * ms) > eps).sum(dim=(-1, -2)) / ms.sum(
                dim=(-1, -2)
            )
            avg_us_all.append(avg_us)
            non_zero_active_us_all.append(non_zero_active_us)

            active_timesteps = ms.any(dim=-1)  # (B, T)
            # For max unmask probability, we aggregate over the block dimension only
            # and then avg over the active timesteps
            max_us = torch.amax(us * ms, dim=(-1))
            max_us = (max_us * active_timesteps).sum(dim=-1) / active_timesteps.sum(
                dim=-1
            )
            max_us_all.append(max_us)
            # For non-zero bs, we take any over the block dimension and
            # then avg over the active timesteps
            non_zero_bs_timesteps = bs.any(dim=-1)  # (B, T)
            non_zero_bs_timesteps = (non_zero_bs_timesteps * active_timesteps).sum(
                dim=-1
            ) / active_timesteps.sum(dim=-1)
            non_zero_bs_timesteps_all.append(non_zero_bs_timesteps)

        avg_us_all = torch.cat(avg_us_all, dim=0)
        max_us_all = torch.cat(max_us_all, dim=0)
        non_zero_active_us_all = torch.cat(non_zero_active_us_all, dim=0)
        non_zero_bs_timesteps_all = torch.cat(non_zero_bs_timesteps_all, dim=0)

        avg_us_all = self.accelerator.gather_for_metrics(avg_us_all)
        non_zero_active_us_all = self.accelerator.gather_for_metrics(
            non_zero_active_us_all
        )
        max_us_all = self.accelerator.gather_for_metrics(max_us_all)

        num_steps = self.accelerator.gather_for_metrics(num_steps).float()

        num_steps = num_steps[post_gathering_policy_only_index]

        non_zero_bs_timesteps_all = self.accelerator.gather_for_metrics(
            non_zero_bs_timesteps_all
        )

        self._metrics[mode]["mean_unmask_prob"].append(avg_us_all.mean().item())
        self._metrics[mode]["non_zero_unmask_prob"].append(
            non_zero_active_us_all.mean().item()
        )
        self._metrics[mode]["max_unmask_prob"].append(max_us_all.mean().item())

        self._metrics[mode]["num_steps_mean"].append(num_steps.mean().item())
        self._metrics[mode]["num_steps_std"].append(num_steps.std().item())
        self._metrics[mode]["num_steps_min"].append(num_steps.min().item())
        self._metrics[mode]["num_steps_max"].append(num_steps.max().item())

        self._metrics[mode]["non_zero_bs_timesteps_mean"].append(
            non_zero_bs_timesteps_all.mean().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_std"].append(
            non_zero_bs_timesteps_all.std().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_min"].append(
            non_zero_bs_timesteps_all.min().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_max"].append(
            non_zero_bs_timesteps_all.max().item()
        )

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if _rich_available:
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # clear cuda memory
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "policy_outputs": policy_outputs_all,
        }
