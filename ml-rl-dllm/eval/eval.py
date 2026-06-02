#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from common.parsing.parse_and_get_acc import check_gsm_correct
from common.parsing.parse_and_get_acc import check_math_correct
from common.parsing.parse_and_get_acc import extract_gsm_answer
from common.parsing.parse_and_get_acc import extract_math_answer
from common.policy_features import get_policy_extra_feature_names
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.humaneval import HumanEvalDataset
from data.loaders.math500 import MATH500Dataset
from data.loaders.mbpp import MBPPDataset
from data.sanitize import sanitize_humaneval
from data.sanitize import sanitize_mbpp

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}

FEW_SHOT_DEFAULTS = {
    "gsm8k": 0,  # NOTE: Fast-dLLM uses 5
    "math": 0,  # NOTE: Fast-dLLM uses 4
    "humaneval": 0,
    "mbpp": 3,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_baseline_checkpoint(name):
    name = name.replace("checkpoint-", "")
    if not name.startswith("baseline-"):
        return None

    params = {"method": name.split("-")[1]}

    # Extract K<number> (tokens per step)
    if match := re.search(r"K(\d+)", name):
        params["diffusion_steps"] = int(match.group(1))

    # Extract t<number> (threshold)
    if match := re.search(r"t([\d.]+)", name):
        params["thres"] = float(match.group(1))

    return params


def evaluate(
    model,
    tokenizer,
    dataloader,
    dataset_name,
    accelerator=None,
    policy=None,
    gen_length=128,
    temperature=0.0,
    steps=64,
    block_length=32,
    remasking="low_confidence",
    thres=0.7,
    sampling_mode="bernoulli",
    dpls_stop_logit=0.0,
    temperature_policy=1.0,
    policy_full_context=True,
    confidences_top_p=1,
    mask_id=126336,
    model_type=None,
    enable_verifier=False,
    verifier_type="math",
    verifier_schedule="final_only",
    verifier_every_k_steps=8,
    verifier_threshold=0.7,
    enable_remasking=False,
    remask_strategy="none",
    max_remasks_per_sample=1,
    remask_cooldown_steps=2,
    min_steps_before_remask=0,
    remask_answer_span_mode="numeric_only",
    target_budget=None,
    method_name=None,
    log_step_traces=False,
    policy_extra_feature_names=None,
    seed=None,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    is_code_dataset = dataset_name in ["humaneval", "mbpp"]

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            disable=(not accelerator.is_main_process if accelerator else False),
        ):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)

            attn_masks = batch["attention_mask"].bool().to(device)
            prompts = batch["prompts"]

            if is_code_dataset:
                if dataset_name == "humaneval":
                    raw_prompts = batch["raw_prompts"]
                    task_ids = batch["task_ids"]
                    test_cases = batch["test_cases"]
                    entry_points = batch["entry_points"]
                elif dataset_name == "mbpp":
                    raw_prompts = batch["texts"]
                    task_ids = batch["task_ids"]
                    test_cases = batch["test_cases"]
                    entry_points = [None] * len(task_ids)
            else:
                gt_answers = batch["answers"]
                questions = batch["questions"]

            gen_kwargs = {
                "model": model,
                "prompt": input_ids,
                "remasking": remasking,
                "gen_length": gen_length,
                "block_length": block_length,
                "temperature": temperature,
                "mask_id": mask_id,
                "model_type": model_type,
                "attention_mask": attn_masks,
                "tokenizer": tokenizer if enable_verifier else None,
                "enable_verifier": enable_verifier,
                "verifier_type": verifier_type,
                "verifier_schedule": verifier_schedule,
                "verifier_every_k_steps": verifier_every_k_steps,
                "verifier_threshold": verifier_threshold,
                "enable_remasking": enable_remasking,
                "remask_strategy": remask_strategy,
                "max_remasks_per_sample": max_remasks_per_sample,
                "remask_cooldown_steps": remask_cooldown_steps,
                "min_steps_before_remask": min_steps_before_remask,
                "remask_answer_span_mode": remask_answer_span_mode,
                "target_budget": target_budget,
                "policy_extra_feature_names": policy_extra_feature_names or [],
                "log_step_traces": log_step_traces,
            }

            if remasking == "policy":
                if policy is None:
                    raise ValueError(
                        "policy remasking requires a policy to be provided"
                    )
                gen_kwargs.update(
                    {
                        "policy": policy,
                        "sampling_mode": sampling_mode,
                        "dpls_stop_logit": dpls_stop_logit,
                        "temperature_policy": temperature_policy,
                        "full_context": policy_full_context,
                        "confidences_top_p": confidences_top_p,
                    }
                )
            elif remasking == "fastdllm":
                gen_kwargs["thres"] = thres
            else:
                gen_kwargs["steps"] = steps

            result = generate_unified(**gen_kwargs)
            out = result.sequences

            steps_taken = [
                int(x) for x in result.steps_taken.detach().cpu().tolist()
            ]

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=True
            )

            batch_wall_time = time.time() - start_time
            wall_time_per_sample = batch_wall_time / len(generated_texts)

            if is_code_dataset:
                sanitized_completions = []
                for j, gen_text in enumerate(generated_texts):
                    if dataset_name == "humaneval":
                        try:
                            full_completion = raw_prompts[j] + gen_text
                            sanitized = sanitize_humaneval(
                                full_completion, entry_points[j]
                            )
                            sanitized_completions.append(sanitized)
                        except Exception as e:
                            print(
                                f"Warning: Failed to sanitize HumanEval completion for {task_ids[j]}: {e}"
                            )
                            # for HumanEval, fall back to just doing prompt + generation
                            sanitized_completions.append(raw_prompts[j] + gen_text)
                    elif dataset_name == "mbpp":
                        try:
                            sanitized = sanitize_mbpp(gen_text)
                            sanitized_completions.append(sanitized)
                        except Exception as e:
                            print(
                                f"Warning: Failed to sanitize MBPP completion for {task_ids[j]}: {e}"
                            )
                            # for MBPP, fall back to just doing the generation
                            sanitized_completions.append(gen_text)

                example_result = [
                    {
                        "task_id": task_ids[j],
                        "prompt": raw_prompts[j],
                        "prompt_input": prompts[j],
                        "generation_raw": generated_texts[j],
                        "generation_sanitized": sanitized_completions[j],
                        "test_cases": test_cases[j],
                        "entry_point": entry_points[j],
                        "steps": steps_taken[j].item()
                        if hasattr(steps_taken[j], "item")
                        else steps_taken[j],
                        "wall_time": wall_time_per_sample,
                    }
                    for j in range(len(task_ids))
                ]

            else:
                example_result = []
                metadata = result.metadata or [{} for _ in range(len(gt_answers))]
                for j in range(len(gt_answers)):
                    ground_truth = (
                        gt_answers[j].item()
                        if hasattr(gt_answers[j], "item")
                        else gt_answers[j]
                    )
                    generation = generated_texts[j]
                    if dataset_name == "gsm8k":
                        final_parsed = extract_gsm_answer(generation)
                        correct_final = check_gsm_correct(final_parsed, ground_truth)
                        before_parsed = (
                            extract_gsm_answer(
                                metadata[j].get("generated_text_before_remask", "")
                            )
                            if metadata[j].get("generated_text_before_remask")
                            else None
                        )
                        after_parsed = (
                            extract_gsm_answer(
                                metadata[j].get("generated_text_after_remask", "")
                            )
                            if metadata[j].get("generated_text_after_remask")
                            else None
                        )
                        correct_before = (
                            check_gsm_correct(before_parsed, ground_truth)
                            if before_parsed is not None
                            else None
                        )
                        correct_after = (
                            check_gsm_correct(after_parsed, ground_truth)
                            if after_parsed is not None
                            else None
                        )
                    else:
                        final_parsed = extract_math_answer(generation)
                        correct_final = check_math_correct(final_parsed, ground_truth)
                        before_parsed = (
                            extract_math_answer(
                                metadata[j].get("generated_text_before_remask", "")
                            )
                            if metadata[j].get("generated_text_before_remask")
                            else None
                        )
                        after_parsed = (
                            extract_math_answer(
                                metadata[j].get("generated_text_after_remask", "")
                            )
                            if metadata[j].get("generated_text_after_remask")
                            else None
                        )
                        correct_before = (
                            check_math_correct(before_parsed, ground_truth)
                            if before_parsed is not None
                            else None
                        )
                        correct_after = (
                            check_math_correct(after_parsed, ground_truth)
                            if after_parsed is not None
                            else None
                        )

                    item = {
                        "example_id": int(total_processed.item()) + j,
                        "dataset": dataset_name,
                        "question": questions[j],
                        "prompt": prompts[j],
                        "prompt_input": prompts[j],
                        "reference_answer": ground_truth,
                        "ground_truth": ground_truth,
                        "generations": generation,
                        "generated_text_before_remask": metadata[j].get(
                            "generated_text_before_remask"
                        ),
                        "generated_text_after_remask": metadata[j].get(
                            "generated_text_after_remask"
                        ),
                        "generated_text_final": generation,
                        "parsed_answer": metadata[j].get("parsed_answer", final_parsed),
                        "correct_before_remask": correct_before,
                        "correct_after_remask": correct_after,
                        "correct_final": correct_final,
                        "steps": steps_taken[j],
                        "nfe_used": metadata[j].get("nfe_used", steps_taken[j]),
                        "actual_nfe": metadata[j].get("actual_nfe", steps_taken[j]),
                        "wall_time": wall_time_per_sample,
                        "target_budget": metadata[j].get("target_budget", target_budget),
                        "budget_error": metadata[j].get("budget_error"),
                        "remask_count": metadata[j].get("remask_count", 0),
                        "remasked_token_count": metadata[j].get(
                            "remasked_token_count", 0
                        ),
                        "verifier_calls": metadata[j].get("verifier_calls", 0),
                        "final_verifier_score": metadata[j].get(
                            "final_verifier_score"
                        ),
                        "verifier_score": metadata[j].get("final_verifier_score"),
                        "parse_ok": metadata[j].get("parse_ok"),
                        "format_ok": metadata[j].get("format_ok"),
                        "arithmetic_ok": metadata[j].get("arithmetic_ok"),
                        "answer_span_text": metadata[j].get("answer_span_text"),
                        "answer_span_token_indices": metadata[j].get(
                            "answer_span_token_indices", []
                        ),
                        "answer_span_mean_confidence": metadata[j].get(
                            "answer_span_mean_confidence"
                        ),
                        "answer_span_min_confidence": metadata[j].get(
                            "answer_span_min_confidence"
                        ),
                        "mean_confidence": metadata[j].get("mean_confidence"),
                        "min_confidence": metadata[j].get("min_confidence"),
                        "entropy_stats": metadata[j].get("entropy_stats"),
                        "stop_reason": metadata[j].get("stop_reason"),
                        "seed": seed,
                        "method_name": method_name or remasking,
                        "block_length": block_length,
                        "max_steps": gen_length,
                        "verifier_result": metadata[j].get("verifier_result"),
                    }
                    if log_step_traces and result.step_traces is not None:
                        item["step_trace"] = result.step_traces[j]
                    example_result.append(item)
            all_generations.extend(example_result)
            total_processed += len(generated_texts)
            wall_times.append(batch_wall_time)

            if accelerator and accelerator.is_main_process:
                idx = random.randint(0, len(prompts) - 1)
                if is_code_dataset:
                    if dataset_name == "humaneval":
                        print(f"Task ID: {task_ids[idx]}")
                        print("-" * 50)
                        print("Generation (sanitized):")
                        print(sanitized_completions[idx])
                        print("-" * 50)
                    elif dataset_name == "mbpp":
                        print(f"Task: {raw_prompts[idx]}")
                        print("-" * 50)
                        print("Generation (sanitized):")
                        print(sanitized_completions[idx])
                        print("-" * 50)
                else:
                    print(f"Question: {questions[idx]}")
                    print("-" * 50)
                    print("Generation:")
                    print(generated_texts[idx])
                    print("-" * 50)
                    print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    avg_nfe = (
        sum(item.get("nfe_used", item.get("steps", 0)) for item in all_generations)
        / len(all_generations)
        if all_generations
        else 0.0
    )
    avg_verifier_calls = (
        sum(item.get("verifier_calls", 0) for item in all_generations)
        / len(all_generations)
        if all_generations
        else 0.0
    )
    avg_remask_count = (
        sum(item.get("remask_count", 0) for item in all_generations)
        / len(all_generations)
        if all_generations
        else 0.0
    )
    parseable_rate = (
        sum(1 for item in all_generations if item.get("parse_ok") is True)
        / len(all_generations)
        if all_generations
        else 0.0
    )
    metrics = {
        "wall_time": avg_wall_time,
        "avg_nfe": avg_nfe,
        "avg_verifier_calls": avg_verifier_calls,
        "avg_remask_count": avg_remask_count,
        "parseable_rate": parseable_rate,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def evaluate_code(generations, dataset_name):
    try:
        print(f"\n=== Running code evaluation for {dataset_name} ===")
        code_eval = hf_evaluate.load("code_eval")

        predictions = [[gen["generation_sanitized"]] for gen in generations]
        references = [gen["test_cases"] for gen in generations]

        print(f"Evaluating {len(predictions)} code samples...")
        pass_at_k, results = code_eval.compute(
            references=references, predictions=predictions, k=[1]
        )
        pass_at_1 = pass_at_k["pass@1"]

        print("Code evaluation results:")
        print(f"  pass@1: {pass_at_1:.4f}")

        for task_id, task_results in results.items():
            if len(task_results) > 0:
                _, result_dict = task_results[0]
                generations[task_id]["pass@1"] = 1.0 if result_dict["passed"] else 0.0
            else:
                generations[task_id]["pass@1"] = 0.0

        return {"pass@1": pass_at_1}

    except Exception as e:
        print(f"Error during code evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_local_path_and_save_results(
    results: dict,
    args: argparse.Namespace,
    model_name: str,
) -> Path | None:
    file_path = None
    if not args.dont_save:
        filename_parts = [
            args.dataset,
            model_name,
            args.gen_length,
            args.diffusion_steps,
            args.block_length,
            args.remasking,
            0,  # for legacy reasons we include the rank of the process
            "generations",
        ]
        file_path = Path(args.output_dir) / (
            "_".join(map(str, filename_parts)) + ".json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=False)
        print(f"Saved results locally to {file_path}")

        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results.get("metrics", {}), f, indent=2, sort_keys=False)

        per_example_path = Path(args.output_dir) / "per_example.jsonl"
        generations_path = Path(args.output_dir) / "generations.jsonl"
        verifier_path = Path(args.output_dir) / "verifier_results.jsonl"
        traces_path = Path(args.output_dir) / "per_step_traces.jsonl"
        with open(per_example_path, "w") as per_example_file, open(
            generations_path, "w"
        ) as generations_file, open(verifier_path, "w") as verifier_file, open(
            traces_path, "w"
        ) as traces_file:
            for item in results.get("generations", []):
                per_example_file.write(json.dumps(item, sort_keys=False) + "\n")
                generations_file.write(
                    json.dumps(
                        {
                            "example_id": item.get("example_id"),
                            "dataset": item.get("dataset", args.dataset),
                            "prompt": item.get("prompt", item.get("prompt_input")),
                            "generated_text_final": item.get(
                                "generated_text_final",
                                item.get("generations", item.get("generation_raw")),
                            ),
                            "reference_answer": item.get(
                                "reference_answer", item.get("ground_truth")
                            ),
                        },
                        sort_keys=False,
                    )
                    + "\n"
                )
                if item.get("verifier_result") is not None:
                    verifier_file.write(
                        json.dumps(
                            {
                                "example_id": item.get("example_id"),
                                "dataset": item.get("dataset", args.dataset),
                                **item["verifier_result"],
                            },
                            sort_keys=False,
                        )
                        + "\n"
                    )
                if item.get("step_trace") is not None:
                    for step_item in item["step_trace"]:
                        traces_file.write(
                            json.dumps(
                                {
                                    "example_id": item.get("example_id"),
                                    "dataset": item.get("dataset", args.dataset),
                                    **step_item,
                                },
                                sort_keys=False,
                            )
                            + "\n"
                        )

        if hasattr(args, "grpo_config"):
            config_path = Path(args.output_dir) / "config_copy.json"
            config_dict = {
                key: str(value)
                if not isinstance(value, (str, int, float, bool, list, dict, type(None)))
                else value
                for key, value in vars(args.grpo_config).items()
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
    return file_path


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument("--model_path", type=str, required=False, default=None)
    parser.add_argument(
        "--few_shot",
        type=int,
        default=-1,
        help="Number of few-shot examples (default: -1 -> dataset-specific defaults)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "humaneval", "mbpp"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=0)
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--remasking", type=str, default="policy")
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--thres", type=float, default=0.7)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature_policy", type=float, default=1.0)
    parser.add_argument("--enable_verifier", action="store_true")
    parser.add_argument("--verifier_type", type=str, default=None)
    parser.add_argument("--verifier_schedule", type=str, default=None)
    parser.add_argument("--verifier_every_k_steps", type=int, default=None)
    parser.add_argument("--verifier_threshold", type=float, default=None)
    parser.add_argument("--enable_remasking", action="store_true")
    parser.add_argument("--remask_strategy", type=str, default=None)
    parser.add_argument("--max_remasks_per_sample", type=int, default=None)
    parser.add_argument("--remask_cooldown_steps", type=int, default=None)
    parser.add_argument("--min_steps_before_remask", type=int, default=None)
    parser.add_argument("--remask_answer_span_mode", type=str, default=None)
    parser.add_argument("--target_budget", type=int, default=None)
    parser.add_argument("--log_step_traces", action="store_true")
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=None,
        help="Sampling mode override (optional, uses config value if not specified)",
    )
    args = parser.parse_args()

    init_seed(args.seed)

    baseline_mode = False
    baseline_params = None
    if args.policy_path:
        checkpoint_name = Path(args.policy_path).parent.name
        baseline_params = parse_baseline_checkpoint(checkpoint_name)
        if baseline_params:
            baseline_mode = True
            print(f"Auto-detected baseline: {baseline_params}")

    # Load args from teh config (unless overriden)
    trl_parser = TrlParser((Config,))
    (grpo_config,) = trl_parser.parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    args.grpo_config = grpo_config
    if args.sampling_mode is None:
        args.sampling_mode = grpo_config.sampling_mode
    if args.block_length is None:
        args.block_length = grpo_config.block_length
    if args.gen_length is None:
        args.gen_length = grpo_config.max_completion_length
    # Override model_path from config if not explicitly provided
    if args.model_path is None:
        args.model_path = grpo_config.model_path
    args.dpls_stop_logit = grpo_config.dpls_stop_logit
    args.enable_verifier = args.enable_verifier or grpo_config.enable_verifier
    args.verifier_type = args.verifier_type or grpo_config.verifier_type
    args.verifier_schedule = args.verifier_schedule or grpo_config.verifier_schedule
    args.verifier_every_k_steps = (
        args.verifier_every_k_steps or grpo_config.verifier_every_k_steps
    )
    args.verifier_threshold = (
        args.verifier_threshold
        if args.verifier_threshold is not None
        else grpo_config.verifier_threshold
    )
    args.enable_remasking = args.enable_remasking or grpo_config.enable_remasking
    args.remask_strategy = args.remask_strategy or grpo_config.remask_strategy
    args.max_remasks_per_sample = (
        args.max_remasks_per_sample or grpo_config.max_remasks_per_sample
    )
    args.remask_cooldown_steps = (
        args.remask_cooldown_steps or grpo_config.remask_cooldown_steps
    )
    args.min_steps_before_remask = (
        args.min_steps_before_remask
        if args.min_steps_before_remask is not None
        else grpo_config.min_steps_before_remask
    )
    args.remask_answer_span_mode = (
        args.remask_answer_span_mode or grpo_config.remask_answer_span_mode
    )
    args.target_budget = args.target_budget or grpo_config.target_budget
    args.log_step_traces = args.log_step_traces or grpo_config.log_step_traces

    if args.remasking == "fastdllm":
        assert args.thres is not None, "thres must be provided for fastdllm"

    # NOTE: setting up the accelerator must be done after parsing config
    accelerator = Accelerator()

    # Check if we are running a baseline, if so get the args from the name
    args.baseline_mode = baseline_mode
    if baseline_mode:
        assert baseline_params is not None
        args.remasking = baseline_params["method"]
        if "thres" in baseline_params:
            args.thres = baseline_params["thres"]
        if "diffusion_steps" in baseline_params:
            args.diffusion_steps = baseline_params["diffusion_steps"]

        args.sampling_mode = None
        if args.remasking in {"random", "low_confidence"}:
            assert args.diffusion_steps > 0

    # Set few_shot to dataset-specific default if -1 is specified
    if args.few_shot == -1:
        args.few_shot = FEW_SHOT_DEFAULTS[args.dataset]
        if accelerator.is_main_process:
            print(
                f"Using dataset-specific few-shot setting for {args.dataset}: {args.few_shot}"
            )

    # Compute model name for output path
    model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    # Load the base model and tokenizer
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if "LLaDA" in args.model_path:
        mask_id = MASK_TOKENS_MAP["LLaDA"]
        _model_type = "LLaDA"
    elif "Dream" in args.model_path:
        mask_id = MASK_TOKENS_MAP["Dream"]
        _model_type = "Dream"
    else:
        raise ValueError(f"Model path {args.model_path} not supported")

    # Load the policy
    policy = None
    if args.remasking == "policy" and not args.baseline_mode:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = args.grpo_config
        if config.policy_type == "dit_hidden":
            assert _model_type == "LLaDA", (
                "dit_hidden policy is only supported with LLaDA models, not Dream"
            )
            policy_core = DiTHiddenStatePolicy(
                dllm=model,
                time_embed_dim=config.policy_time_embed_dim,
                num_blocks=config.policy_num_blocks,
                smart_init=config.policy_smart_init,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_confidence":
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)
            policy_extra_feature_names = get_policy_extra_feature_names(config)

            policy_core = DiTConfidencePolicy(
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                confidences_top_p=config.confidences_top_p,
                extra_feature_dim=len(policy_extra_feature_names),
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        else:
            raise ValueError(
                f"Policy type {config.policy_type} not supported. "
                "Choose from ['dit_hidden', 'dit_confidence']"
            )
        policy = PolicyHFWrapper(policy_core, config.policy_type)

        if args.policy_path is not None:
            if accelerator.is_main_process:
                print(f"Loading policy from {args.policy_path}")
            state = load_file(args.policy_path)
            policy.load_state_dict(state)

    # Create the dataset
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": -1,
        "num_examples": args.few_shot,
    }
    if args.dataset in ["gsm8k", "math"]:
        dataset_kwargs["add_reasoning"] = True
    dataset = DATASET_MAP[args.dataset](**dataset_kwargs)

    # take only first args.n_test examples
    collate_fn = dataset.collate_fn
    if args.n_test is not None and len(dataset) > args.n_test:
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=collate_fn,
    )

    # Use accelerator to prepare model and policy, but NOT the dataloader
    # We manage distribution manually with CustomDistributedSampler to avoid padding
    if policy is not None:
        model, policy = accelerator.prepare(model, policy)
    else:
        model = accelerator.prepare(model)

    # Run evaluation
    results = evaluate(
        model,
        tokenizer,
        dataloader,
        dataset_name=args.dataset,
        accelerator=accelerator,
        policy=policy,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        remasking=args.remasking,
        thres=args.thres,
        sampling_mode=args.sampling_mode,
        dpls_stop_logit=args.dpls_stop_logit,
        temperature_policy=args.temperature_policy,
        mask_id=mask_id,
        model_type=_model_type,
        policy_full_context=args.grpo_config.policy_full_context
        if args.remasking == "policy"
        else False,
        confidences_top_p=args.grpo_config.confidences_top_p
        if args.remasking == "policy"
        else 1,
        enable_verifier=args.enable_verifier,
        verifier_type=args.verifier_type,
        verifier_schedule=args.verifier_schedule,
        verifier_every_k_steps=args.verifier_every_k_steps,
        verifier_threshold=args.verifier_threshold,
        enable_remasking=args.enable_remasking,
        remask_strategy=args.remask_strategy,
        max_remasks_per_sample=args.max_remasks_per_sample,
        remask_cooldown_steps=args.remask_cooldown_steps,
        min_steps_before_remask=args.min_steps_before_remask,
        remask_answer_span_mode=args.remask_answer_span_mode,
        target_budget=args.target_budget
        if args.grpo_config.enable_budget_conditioning
        else None,
        method_name=args.grpo_config.method,
        log_step_traces=args.log_step_traces,
        policy_extra_feature_names=get_policy_extra_feature_names(args.grpo_config),
        seed=args.seed,
    )

    if accelerator.num_processes > 1:
        all_gpu_generations = gather_object(results["generations"])
        if accelerator.is_main_process:
            results["generations"] = all_gpu_generations

    if accelerator.is_main_process:
        if args.dataset in {"humaneval", "mbpp"}:
            results["code_eval_results"] = evaluate_code(
                results["generations"], args.dataset
            )
        results["metrics"] = {
            k: results.pop(k)
            for k in (
                "wall_time",
                "avg_nfe",
                "avg_verifier_calls",
                "avg_remask_count",
                "parseable_rate",
                "total_processed",
            )
        }
        results.update(
            {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
                "remasking": args.remasking,
                "policy_path": args.policy_path,
                "thres": args.thres,
                "n_test": args.n_test,
                "enable_verifier": args.enable_verifier,
                "verifier_type": args.verifier_type,
                "verifier_schedule": args.verifier_schedule,
                "enable_remasking": args.enable_remasking,
                "remask_strategy": args.remask_strategy,
                "target_budget": args.target_budget,
                "reward_type": args.grpo_config.reward_type,
            }
        )
        get_local_path_and_save_results(results, args, model_name)

        # Before exiting, print some basic metrics about the test set to make sure we processed
        # as many samples as we expected
        actual_samples_processed = len(results["generations"])
        expected_dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        if hasattr(dataset, "dataset"):  # Handle Subset wrapper
            expected_dataset_size = (
                len(dataset.dataset) if args.n_test is None else args.n_test
            )
        elif args.n_test is not None:
            expected_dataset_size = args.n_test

        print("\n=== Test Set Verification ===")
        print(f"Dataset: {args.dataset}")
        print(f"Samples processed: {actual_samples_processed}")
        print(f"Expected dataset size: {expected_dataset_size}")
        if expected_dataset_size:
            print(
                f"Coverage: {actual_samples_processed}/{expected_dataset_size} ({100 * actual_samples_processed / expected_dataset_size:.1f}%)"
            )
        print(f"Batch size: {args.batch_size}")
        print(f"Multi-GPU processes: {accelerator.num_processes}")
        print("=============================\n")

    accelerator.end_training()
    accelerator.free_memory()
