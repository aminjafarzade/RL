#!/usr/bin/env python
"""Run fixed-policy post-hoc continuation analysis without GRPO updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.cuda_env import set_cuda_visible_devices_from_argv


set_cuda_visible_devices_from_argv()

import torch
import yaml


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}
OUTPUT_JSONL_NAMES = (
    "per_example.jsonl",
    "generations.jsonl",
    "verifier_results.jsonl",
    "continuation_records.jsonl",
)


def log(message: str) -> None:
    print(f"[continuation-analysis] {message}", flush=True)


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def cfg(config: dict[str, Any], key: str, default=None):
    return config.get(key, default)


def resolve_policy_checkpoint_file(checkpoint_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(checkpoint_path)))
    if not expanded.exists():
        raise FileNotFoundError(
            f"policy_checkpoint_path does not exist: {checkpoint_path}"
        )
    if expanded.is_file():
        return expanded

    preferred_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "policy.safetensors",
        "policy.pt",
        "policy.bin",
        "checkpoint.pt",
    )
    for name in preferred_names:
        candidate = expanded / name
        if candidate.exists():
            return candidate

    candidates = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        candidates.extend(sorted(expanded.glob(pattern)))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(path.name for path in candidates[:8])
        raise FileNotFoundError(
            "policy_checkpoint_path contains multiple candidate policy files "
            f"without a preferred name: {checkpoint_path}. Candidates: {names}"
        )
    raise FileNotFoundError(
        "policy_checkpoint_path must point to a policy checkpoint file or a "
        "directory containing model.safetensors or pytorch_model.bin: "
        f"{checkpoint_path}"
    )


def load_policy_checkpoint(policy: PolicyHFWrapper, checkpoint_path: str, device):
    log(f"resolving policy checkpoint: {checkpoint_path}")
    checkpoint_file = resolve_policy_checkpoint_file(checkpoint_path)
    log(f"loading lightweight policy checkpoint file: {checkpoint_file}")
    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a .safetensors policy checkpoint requires safetensors."
            ) from exc
        state_dict = load_file(str(checkpoint_file), device="cpu")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Policy checkpoint did not contain a state dict: {checkpoint_file}"
        )

    result = policy.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "Policy checkpoint keys did not match the configured policy. "
            f"missing={missing}, unexpected={unexpected}, path={checkpoint_file}"
        )
    policy.to(device)
    log(f"loaded lightweight policy checkpoint: {checkpoint_file}")
    return {
        "policy_checkpoint_path_effective": str(checkpoint_file),
        "policy_checkpoint_loaded": True,
        "policy_checkpoint_missing_keys": missing,
        "policy_checkpoint_unexpected_keys": unexpected,
    }


def build_policy(config: dict[str, Any], model, device):
    from common.models.policy import DiTConfidencePolicy
    from common.models.policy import DiTHiddenStatePolicy
    from common.models.policy import PolicyHFWrapper
    from common.policy_features import get_policy_extra_feature_names

    policy_type = cfg(config, "policy_type", "dit_confidence")
    log(f"building lightweight policy: policy_type={policy_type}")
    if policy_type == "dit_hidden":
        if cfg(config, "model_type") != "LLaDA":
            raise ValueError("dit_hidden policy is only supported with LLaDA")
        policy_core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=cfg(config, "policy_time_embed_dim", 128),
            num_blocks=cfg(config, "policy_num_blocks", 1),
            smart_init=cfg(config, "policy_smart_init"),
            time_period=cfg(config, "policy_time_period", 1),
        ).to(device)
    elif policy_type == "dit_confidence":
        hidden_dim = cfg(config, "policy_hidden_dim", 128) or 128
        feedforward_dim = cfg(config, "policy_feedforward_dim") or (4 * hidden_dim)
        policy_args = SimpleNamespace(**config)
        extra_names = get_policy_extra_feature_names(policy_args)
        policy_core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=cfg(config, "policy_num_heads", 2),
            dropout=cfg(config, "policy_dropout", 0.0),
            time_embed_dim=cfg(config, "policy_time_embed_dim", 128),
            smart_init=cfg(config, "policy_smart_init"),
            confidences_top_p=cfg(config, "confidences_top_p", 1),
            extra_feature_dim=len(extra_names),
            num_blocks=cfg(config, "policy_num_blocks", 1),
            time_period=cfg(config, "policy_time_period", 1),
        ).to(device)
    else:
        raise ValueError(f"Unsupported policy_type={policy_type}")
    policy = PolicyHFWrapper(policy_core, policy_type).to(device)
    total_params = sum(param.numel() for param in policy.parameters())
    log(f"built lightweight policy with {total_params:,} parameters")
    return policy


def load_model_and_tokenizer(config: dict[str, Any], device):
    try:
        from transformers import AutoModel
        from transformers import AutoTokenizer
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Running continuation analysis requires transformers. "
            "Install the project GPU/runtime dependencies before invoking main()."
        ) from exc

    bnb_config = None
    if cfg(config, "load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model_path = cfg(config, "model_path", "")
    if "LLaDA" in model_path:
        model_type = "LLaDA"
    elif "Dream" in model_path:
        model_type = "Dream"
    else:
        raise ValueError(f"Model path {model_path} not supported")

    log(f"loading frozen base model: {model_path} on device={device}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)
    log("loaded frozen base model")
    log(f"loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    config["model_type"] = model_type
    config["mask_id"] = MASK_TOKENS_MAP[model_type]
    log(f"loaded tokenizer; model_type={model_type}, mask_id={config['mask_id']}")
    return model.eval(), tokenizer


def load_split(config: dict[str, Any], split_name: str, max_samples: int | None):
    from data.data_utils import get_gsm8k_questions

    log(f"loading GSM8K split={split_name}, max_samples={max_samples}")
    dataset = get_gsm8k_questions("test" if split_name == "eval" else "train")
    dataset = dataset.shuffle(seed=int(cfg(config, "seed", 123)))
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    log(f"loaded split={split_name}, samples={len(dataset)}")
    return dataset


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def prompt_text_from_example(example: dict[str, Any], tokenizer) -> str:
    prompt = example.get("prompt", "")
    if isinstance(prompt, list):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return "\n".join(
            str(message.get("content", ""))
            if isinstance(message, dict)
            else str(message)
            for message in prompt
        )
    return str(prompt)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gpu",
        help=(
            "Physical GPU index/list to expose before torch initializes. "
            "Example: --gpu 2 makes physical GPU 2 visible as cuda:0."
        ),
    )
    parser.add_argument(
        "--cuda_visible_devices",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        help="Direct CUDA_VISIBLE_DEVICES value, e.g. '2' or '0,1'.",
    )
    parser.add_argument(
        "--cuda_min_memory_gb",
        "--cuda-min-memory-gb",
        dest="cuda_min_memory_gb",
        type=float,
        help="Auto-select the smallest GPU with at least this much total memory.",
    )
    parser.add_argument("--policy_checkpoint_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_train_index", type=int, default=0)
    parser.add_argument("--start_eval_index", type=int, default=0)
    parser.add_argument("--end_train_index", type=int)
    parser.add_argument("--end_eval_index", type=int)
    parser.add_argument("--skip_failed_batches", action="store_true")
    parser.add_argument("--failed_batches_path", default="failed_batches.jsonl")
    return parser


def resolve_failed_batches_path(output_dir: Path, failed_batches_path: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(failed_batches_path)))
    if path.is_absolute():
        return path
    return output_dir / path


def prepare_output_files(
    output_dir: Path,
    *,
    resume: bool,
    failed_batches_path: Path | None = None,
) -> None:
    if resume:
        log("resume enabled; preserving existing output jsonl files")
        return
    for name in OUTPUT_JSONL_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()
    if failed_batches_path is not None and failed_batches_path.exists():
        failed_batches_path.unlink()


def split_index_range(
    *,
    config: dict[str, Any],
    split: str,
    dataset_len: int,
    resume: bool,
    start_index: int,
    end_index: int | None,
) -> tuple[int, int]:
    if not resume:
        return 0, dataset_len

    max_samples = (
        cfg(config, "max_train_samples")
        if split == "train"
        else cfg(config, "max_eval_samples")
    )
    effective_end = end_index
    if effective_end is None and max_samples is not None:
        effective_end = int(max_samples)
    if effective_end is None:
        effective_end = dataset_len

    start = max(0, int(start_index))
    end = min(max(0, int(effective_end)), dataset_len)
    if start > end:
        start = end
    return start, end


def write_failed_batch(
    *,
    failed_batches_path: Path,
    mode: str,
    start: int,
    end: int,
    error: Exception,
) -> None:
    failed_batches_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": mode,
        "start": start,
        "end": end,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    with failed_batches_path.open("a") as f:
        f.write(json.dumps(jsonable(record)) + "\n")


def quality_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "extractor_mode": cfg(config, "reward_gsm_extractor", "answer_span"),
        "reward_quality_mode": cfg(config, "reward_quality_mode", "format_weighted"),
        "answer_span_partial_credit": cfg(config, "answer_span_partial_credit", 0.45),
        "malformed_answer_factor": cfg(config, "malformed_answer_factor", 0.5),
        "enable_repetition_penalty": cfg(config, "enable_repetition_penalty", True),
        "repetition_unique_ratio_threshold": cfg(
            config,
            "repetition_unique_ratio_threshold",
            0.25,
        ),
        "repetition_max_freq_threshold": cfg(
            config,
            "repetition_max_freq_threshold",
            0.25,
        ),
        "repetition_penalty_factor": cfg(config, "repetition_penalty_factor", 0.5),
    }


def base_reward_fields(
    *,
    text: str,
    answer,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    from common.parsing.parse_and_get_acc import extract_gsm_answer_with_source
    from common.parsing.parse_and_get_acc import numeric_equal
    from common.rewards import apply_reward_quality_caps
    from common.rewards import compute_budget_reward
    from common.rewards import compute_reward_quality

    selected = extract_gsm_answer_with_source(
        text,
        mode=cfg(config, "reward_gsm_extractor", "answer_span"),
    )
    quality = compute_reward_quality(
        text,
        answer,
        format_ok=metadata.get("format_ok"),
        **quality_kwargs(config),
    )
    budget = compute_budget_reward(
        task_score=quality.task_score,
        nfe_used=metadata.get("actual_nfe", metadata.get("nfe_used", 0)),
        target_budget=metadata.get("target_budget", cfg(config, "target_budget", 1)),
        max_steps=metadata.get("max_steps", cfg(config, "max_completion_length", 1)),
        remask_count=metadata.get("remask_count", 0),
        reward_budget_mode=cfg(config, "reward_budget_mode", "threshold"),
        reward_beta=cfg(config, "reward_beta", 2.0),
        reward_mu=cfg(config, "reward_mu", 0.0),
        reward_rho=cfg(config, "reward_rho", 0.0),
        threshold_reward_hard_zero=cfg(config, "threshold_reward_hard_zero", False),
    )
    is_clean_format = bool(quality.strict_correct) or bool(metadata.get("format_ok"))
    repeated_junk = bool(quality.repeated_char_run_detected) or (
        quality.repetition_penalty < 1.0
    )
    cap = apply_reward_quality_caps(
        reward=budget.reward,
        is_malformed=not is_clean_format,
        repeated_junk_detected=repeated_junk,
        max_reward_if_malformed=cfg(config, "max_reward_if_malformed"),
        max_reward_if_repetitive=cfg(config, "max_reward_if_repetitive"),
    )
    return {
        "base_parsed_answer": quality.parsed_answer,
        "base_task_score": quality.task_score,
        "base_correct": quality.task_score > 0.0,
        "base_quality_score": quality.task_score,
        "base_reward_answer_source": selected.source,
        "base_repetition_penalty": quality.repetition_penalty,
        "base_answer_quality": quality.answer_quality,
        "base_format_factor": quality.format_factor,
        "strict_correct": quality.strict_correct,
        "answer_span_correct": quality.answer_span_correct,
        "robust_correct": quality.robust_correct,
        "parsed_answer_used_for_reward": quality.parsed_answer,
        "reward_answer_source": selected.source,
        "reward_exact_match_used": numeric_equal(selected.answer, answer),
        "answer_quality": quality.answer_quality,
        "format_factor": quality.format_factor,
        "repetition_penalty": quality.repetition_penalty,
        "unique_token_ratio": quality.unique_token_ratio,
        "max_token_freq_ratio": quality.max_token_freq_ratio,
        "repeated_char_run_detected": quality.repeated_char_run_detected,
        "task_score_after_quality": quality.task_score,
        "final_task_score": quality.task_score,
        "reward": cap.reward_after_quality_caps,
        "reward_before_quality_caps": cap.reward_before_quality_caps,
        "reward_after_quality_caps": cap.reward_after_quality_caps,
        "malformed_reward_cap_applied": cap.malformed_reward_cap_applied,
        "repetitive_reward_cap_applied": cap.repetitive_reward_cap_applied,
        "threshold_gate": budget.threshold_gate,
        "budget_over_penalty": budget.budget_over_penalty,
        "nfe_target_ratio": budget.nfe_target_ratio,
    }


def enrich_continuations(
    *,
    continuation_results: list[dict[str, Any]],
    base_task_score: float,
    base_correct: bool,
    answer,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from common.generation.generation import continuation_metric_fields
    from common.parsing.parse_and_get_acc import extract_gsm_answer_with_source
    from common.rewards import compute_reward_quality

    enriched = []
    for item in continuation_results:
        text = item.get("continued_generated_text") or ""
        quality = compute_reward_quality(
            text,
            answer,
            format_ok=item.get("continued_format_ok"),
            **quality_kwargs(config),
        )
        source = extract_gsm_answer_with_source(
            text,
            mode=cfg(config, "reward_gsm_extractor", "answer_span"),
        ).source
        continued_correct = quality.task_score > 0.0
        enriched.append(
            {
                **item,
                "continuation_extra_steps": item.get("continued_extra_steps"),
                "continued_parsed_answer": quality.parsed_answer,
                "continued_task_score": quality.task_score,
                "continued_correct": continued_correct,
                "continued_quality_score": quality.task_score,
                "continued_reward_answer_source": source,
                **continuation_metric_fields(
                    base_task_score=base_task_score,
                    continued_task_score=quality.task_score,
                    base_correct=base_correct,
                    continued_correct=continued_correct,
                    base_quality_score=base_task_score,
                    continued_quality_score=quality.task_score,
                ),
            }
        )
    return enriched


def build_records_for_batch(
    *,
    examples,
    prompts_text,
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    checkpoint_info: dict[str, Any],
    example_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_example = []
    continuation_records = []
    verifier_records = []
    if example_indices is None:
        example_indices = list(range(len(examples)))
    for idx, (example, prompt_text, meta) in enumerate(
        zip(examples, prompts_text, metadata)
    ):
        example_index = example_indices[idx]
        text = meta.get("generated_text_final") or ""
        base_fields = base_reward_fields(
            text=text,
            answer=example["answer"],
            metadata=meta,
            config=config,
        )
        continuations = enrich_continuations(
            continuation_results=meta.get("continuation_results") or [],
            base_task_score=base_fields["base_task_score"],
            base_correct=base_fields["base_correct"],
            answer=example["answer"],
            config=config,
        )
        best = None
        if continuations:
            best = max(
                continuations,
                key=lambda item: (
                    float(item.get("continued_task_score") or 0.0),
                    -int(item.get("continuation_k") or 0),
                ),
            )
        target_budget = int(meta.get("target_budget", cfg(config, "target_budget", 0)))
        base_nfe = int(meta.get("base_nfe", meta.get("actual_nfe", 0)))
        base_span = meta.get("base_answer_span") or {}
        row = {
            **meta,
            **base_fields,
            **checkpoint_info,
            "mode": mode,
            "example_id": f"{mode}-{example_index}-{prompt_hash(prompt_text)}",
            "example_index": example_index,
            "prompt_hash": prompt_hash(prompt_text),
            "dataset": example.get("dataset_type", cfg(config, "dataset", "gsm8k")),
            "prompt": prompt_text,
            "reference_answer": example.get("answer"),
            "target_budget": target_budget,
            "base_nfe": base_nfe,
            "remaining_budget": target_budget - base_nfe,
            "base_generated_text": text,
            "generated_text": text,
            "generated_text_final": text,
            "base_format_ok": meta.get("base_format_ok", meta.get("format_ok")),
            "base_parse_ok": meta.get("base_parse_ok", meta.get("parse_ok")),
            "base_arithmetic_ok": meta.get(
                "base_arithmetic_ok",
                meta.get("arithmetic_ok"),
            ),
            "base_verifier_score": meta.get(
                "base_verifier_score",
                meta.get("final_verifier_score"),
            ),
            "base_visible_generated_char_count": len(text),
            "visible_generated_char_count": len(text),
            "base_answer_span_text": meta.get("answer_span_text"),
            "base_answer_span_token_indices": base_span.get(
                "token_indices",
                meta.get("answer_span_token_indices", []),
            ),
            "base_answer_span_confidence": meta.get("base_answer_span_confidence"),
            "base_entropy_stats": meta.get("entropy_stats"),
            "base_confidence_stats": {
                "mean": meta.get("mean_confidence"),
                "min": meta.get("min_confidence"),
            },
            "continuation_results": continuations,
            "best_continuation_k": best.get("continuation_k") if best else None,
            "best_continuation_task_score": best.get("continued_task_score")
            if best
            else None,
            "best_continuation_gain": best.get("continuation_gain") if best else None,
            "best_continuation_correct": best.get("continued_correct")
            if best
            else None,
            "best_continuation_text": best.get("continued_generated_text")
            if best
            else None,
            "oracle_should_continue": bool(
                best and float(best.get("continuation_gain") or 0.0) > 0.0
            ),
            "num_generations": cfg(config, "num_generations", 1),
            "max_completion_length": cfg(config, "max_completion_length", 192),
            "block_length": cfg(config, "block_length", 32),
            "generation_batch_size": cfg(config, "generation_batch_size", 8),
            "per_device_train_batch_size": cfg(config, "per_device_train_batch_size", 1),
            "policy_smart_init": cfg(config, "policy_smart_init"),
            "policy_checkpoint_path": checkpoint_info.get(
                "policy_checkpoint_path_effective"
            ),
        }
        per_example.append(row)
        if meta.get("verifier_result") is not None:
            verifier_records.append(
                {
                    "example_id": row["example_id"],
                    "mode": mode,
                    **(meta.get("verifier_result") or {}),
                }
            )
        for item in continuations:
            continuation_records.append(
                {
                    **{k: v for k, v in row.items() if k != "continuation_results"},
                    **item,
                }
            )
    return per_example, continuation_records, verifier_records


def target_budget_for_index(config: dict[str, Any], mode: str, index: int) -> int:
    choices = (
        cfg(config, "train_budget_sampling")
        if mode == "train"
        else cfg(config, "target_budgets")
    ) or [cfg(config, "target_budget", 32)]
    return int(choices[index % len(choices)])


def run_split(
    *,
    config: dict[str, Any],
    mode: str,
    dataset,
    model,
    policy,
    tokenizer,
    output_dir: Path,
    checkpoint_info: dict[str, Any],
    start_index: int = 0,
    end_index: int | None = None,
    skip_failed_batches: bool = False,
    failed_batches_path: Path | None = None,
):
    from common.generation.generation import generate_unified
    from common.policy_features import get_policy_extra_feature_names

    generation_batch_size = int(cfg(config, "generation_batch_size", 8) or 8)
    device = next(policy.parameters()).device
    per_example_path = output_dir / "per_example.jsonl"
    generations_path = output_dir / "generations.jsonl"
    verifier_path = output_dir / "verifier_results.jsonl"
    continuation_path = output_dir / "continuation_records.jsonl"
    if end_index is None:
        end_index = len(dataset)
    log(
        f"{mode}: processing index range [{start_index}, {end_index}) "
        f"of {len(dataset)} samples"
    )

    with per_example_path.open("a") as per_f, continuation_path.open("a") as cont_f:
        gen_f = generations_path.open("a") if cfg(config, "save_generations", True) else None
        ver_f = verifier_path.open("a") if cfg(config, "save_verifier_results", True) else None
        try:
            for start in range(start_index, end_index, generation_batch_size):
                end = min(start + generation_batch_size, end_index)
                log(
                    f"{mode}: starting batch {start // generation_batch_size + 1} "
                    f"examples {start}-{end}"
                )
                try:
                    examples = [dataset[i] for i in range(start, end)]
                    prompts_text = [
                        prompt_text_from_example(example, tokenizer)
                        for example in examples
                    ]
                    prompt_inputs = tokenizer(
                        text=prompts_text,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_inputs["input_ids"].to(device)
                    prompt_mask = prompt_inputs["attention_mask"].to(device)
                    max_prompt_length = cfg(config, "max_prompt_length")
                    if max_prompt_length is not None:
                        prompt_ids = prompt_ids[:, -int(max_prompt_length) :]
                        prompt_mask = prompt_mask[:, -int(max_prompt_length) :]
                    target_budgets = torch.tensor(
                        [
                            target_budget_for_index(config, mode, idx)
                            for idx in range(start, end)
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                    with torch.no_grad():
                        result = generate_unified(
                            model=model,
                            prompt=prompt_ids,
                            remasking="policy",
                            policy=policy,
                            gen_length=int(cfg(config, "max_completion_length", 192)),
                            block_length=int(cfg(config, "block_length", 32)),
                            temperature=float(cfg(config, "temperature", 0.0) or 0.0),
                            mask_id=int(cfg(config, "mask_id")),
                            sampling_mode=cfg(config, "sampling_mode", "bernoulli"),
                            dpls_stop_logit=float(cfg(config, "dpls_stop_logit", 0.0)),
                            model_type=cfg(config, "model_type"),
                            attention_mask=prompt_mask,
                            temperature_policy=float(
                                cfg(config, "temperature_policy", 1.0)
                            ),
                            full_context=bool(cfg(config, "policy_full_context", True)),
                            confidences_top_p=int(cfg(config, "confidences_top_p", 1)),
                            tokenizer=tokenizer,
                            enable_verifier=bool(cfg(config, "enable_verifier", True)),
                            verifier_type=cfg(config, "verifier_type", "math"),
                            verifier_schedule=cfg(
                                config,
                                "verifier_schedule",
                                "final_only",
                            ),
                            verifier_every_k_steps=int(
                                cfg(config, "verifier_every_k_steps", 8)
                            ),
                            verifier_threshold=float(
                                cfg(config, "verifier_threshold", 0.7)
                            ),
                            enable_remasking=bool(
                                cfg(config, "enable_remasking", False)
                            ),
                            remask_strategy=cfg(config, "remask_strategy", "none"),
                            max_remasks_per_sample=int(
                                cfg(config, "max_remasks_per_sample", 0)
                            ),
                            remask_answer_span_mode=cfg(
                                config,
                                "remask_answer_span_mode",
                                "numeric_only",
                            ),
                            target_budget=target_budgets
                            if cfg(config, "enable_budget_conditioning", True)
                            else None,
                            hard_stop_at_target_budget=bool(
                                cfg(config, "hard_stop_at_target_budget", False)
                            ),
                            hard_generation_budget=cfg(
                                config,
                                "hard_generation_budget",
                            ),
                            policy_extra_feature_names=get_policy_extra_feature_names(
                                SimpleNamespace(**config)
                            ),
                            log_step_traces=bool(
                                cfg(config, "log_step_traces", False)
                            ),
                            rollout_compute_style=["normal"] * len(examples),
                            patient_global_slowdown=False,
                            patient_delay_low_confidence=False,
                            patient_delay_final_window=False,
                            enable_deliberation_probe=False,
                            enable_posthoc_continuation=bool(
                                cfg(config, "enable_posthoc_continuation", True)
                            ),
                            continuation_steps=list(
                                cfg(config, "continuation_steps", [4])
                            ),
                            continuation_trigger=cfg(
                                config,
                                "continuation_trigger",
                                "always_for_analysis",
                            ),
                            continuation_remask_mode=cfg(
                                config,
                                "continuation_remask_mode",
                                "answer_span",
                            ),
                            continuation_final_window_tokens=int(
                                cfg(config, "continuation_final_window_tokens", 32)
                            ),
                            continuation_low_confidence_quantile=float(
                                cfg(
                                    config,
                                    "continuation_low_confidence_quantile",
                                    0.3,
                                )
                            ),
                            continuation_min_remaining_budget=int(
                                cfg(
                                    config,
                                    "continuation_min_remaining_budget",
                                    4,
                                )
                            ),
                            continuation_use_oracle_for_analysis=bool(
                                cfg(
                                    config,
                                    "continuation_use_oracle_for_analysis",
                                    True,
                                )
                            ),
                            continuation_force_for_analysis=bool(
                                cfg(config, "continuation_force_for_analysis", True)
                            ),
                            continuation_select_best_k_oracle=bool(
                                cfg(
                                    config,
                                    "continuation_select_best_k_oracle",
                                    False,
                                )
                            ),
                            continuation_confidence_threshold=float(
                                cfg(config, "continuation_confidence_threshold", 0.5)
                            ),
                            continuation_verifier_threshold=float(
                                cfg(config, "continuation_verifier_threshold", 0.7)
                            ),
                        )
                    log(
                        f"{mode}: finished generation/continuation for "
                        f"examples {start}-{end}"
                    )
                    metadata = result.metadata or []
                    for local_idx, item in enumerate(metadata):
                        item["target_budget"] = int(target_budgets[local_idx].item())
                        item["budget_error"] = item.get("nfe_used", 0) - item[
                            "target_budget"
                        ]
                        item["rollout_compute_style"] = "normal"
                        item["rollout_index"] = 0
                        item["group_id"] = start + local_idx
                        item["num_generations"] = 1
                    rows, cont_rows, verifier_rows = build_records_for_batch(
                        examples=examples,
                        prompts_text=prompts_text,
                        metadata=metadata,
                        config=config,
                        mode=mode,
                        checkpoint_info=checkpoint_info,
                        example_indices=list(range(start, end)),
                    )
                    for row in rows:
                        per_f.write(json.dumps(jsonable(row)) + "\n")
                        if gen_f is not None:
                            gen_f.write(
                                json.dumps(
                                    jsonable(
                                        {
                                            "example_id": row["example_id"],
                                            "mode": mode,
                                            "generated_text": row[
                                                "base_generated_text"
                                            ],
                                            "target_budget": row["target_budget"],
                                        }
                                    )
                                )
                                + "\n"
                            )
                    for row in cont_rows:
                        cont_f.write(json.dumps(jsonable(row)) + "\n")
                    if ver_f is not None:
                        for row in verifier_rows:
                            ver_f.write(json.dumps(jsonable(row)) + "\n")
                    per_f.flush()
                    cont_f.flush()
                    if gen_f is not None:
                        gen_f.flush()
                    if ver_f is not None:
                        ver_f.flush()
                    log(f"{mode}: wrote examples {start}-{end}")
                except Exception as exc:
                    if not skip_failed_batches:
                        raise
                    if failed_batches_path is None:
                        raise
                    write_failed_batch(
                        failed_batches_path=failed_batches_path,
                        mode=mode,
                        start=start,
                        end=end,
                        error=exc,
                    )
                    log(f"{mode}: skipped failed batch {start}-{end}: {exc}")
        finally:
            if gen_f is not None:
                gen_f.close()
            if ver_f is not None:
                ver_f.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    log(f"starting analysis with config={args.config}")
    config = read_config(args.config)
    if config.get("dataset") != "gsm8k":
        raise ValueError("Continuation analysis currently supports dataset: gsm8k")
    if not config.get("analysis_only", False):
        print("warning: analysis_only is not true in config; no training will run anyway")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_train_samples is not None:
        config["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        config["max_eval_samples"] = args.max_eval_samples
    if args.policy_checkpoint_path:
        config["policy_checkpoint_path"] = args.policy_checkpoint_path

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"output_dir={output_dir}")
    failed_batches_path = resolve_failed_batches_path(
        output_dir,
        args.failed_batches_path,
    )
    prepare_output_files(
        output_dir,
        resume=args.resume,
        failed_batches_path=failed_batches_path
        if args.skip_failed_batches
        else None,
    )
    from data.data_utils import set_random_seed

    set_random_seed(int(cfg(config, "seed", 123)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        log(
            "CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}; "
            f"using process cuda:{current_device} "
            f"({torch.cuda.get_device_name(current_device)})"
        )
    else:
        log(f"using device={device}")

    requested_checkpoint = config.get("policy_checkpoint_path")
    checkpoint_info = {
        "policy_checkpoint_path_requested": requested_checkpoint,
        "policy_checkpoint_path_effective": None,
        "policy_checkpoint_loaded": False,
        "policy_checkpoint_missing_keys": [],
        "policy_checkpoint_unexpected_keys": [],
    }

    model, tokenizer = load_model_and_tokenizer(config, device)
    policy = build_policy(config, model, device)
    if requested_checkpoint:
        checkpoint_info.update(
            load_policy_checkpoint(policy, requested_checkpoint, device)
        )
        log(
            "Loaded policy checkpoint: "
            f"{checkpoint_info['policy_checkpoint_path_effective']}"
        )
    else:
        log("no policy checkpoint requested; using configured policy initialization")
    policy.eval()
    if cfg(config, "freeze_policy", True):
        for param in policy.parameters():
            param.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    with (output_dir / "analysis_config.json").open("w") as f:
        json.dump(jsonable({**config, **checkpoint_info}), f, indent=2)
    log("wrote analysis_config.json")

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for split in splits:
        configured_max_samples = (
            cfg(config, "max_train_samples")
            if split == "train"
            else cfg(config, "max_eval_samples")
        )
        max_samples = None if args.resume else configured_max_samples
        dataset = load_split(config, split, max_samples)
        start_arg = (
            args.start_train_index if split == "train" else args.start_eval_index
        )
        end_arg = args.end_train_index if split == "train" else args.end_eval_index
        start_index, end_index = split_index_range(
            config=config,
            split=split,
            dataset_len=len(dataset),
            resume=args.resume,
            start_index=start_arg,
            end_index=end_arg,
        )
        log(
            f"{split} samples: {len(dataset)}; processing indices "
            f"[{start_index}, {end_index})"
        )
        run_split(
            config=config,
            mode=split,
            dataset=dataset,
            model=model,
            policy=policy,
            tokenizer=tokenizer,
            output_dir=output_dir,
            checkpoint_info=checkpoint_info,
            start_index=start_index,
            end_index=end_index,
            skip_failed_batches=args.skip_failed_batches,
            failed_batches_path=failed_batches_path,
        )
    log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
