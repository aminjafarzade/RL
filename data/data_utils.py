#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk

from common.parsing.parser_utils import extract_hash_answer

LOCAL_DATASETS_DIR = Path(os.environ.get("RLDLLM_DATASETS_DIR", "/mnt/datasets"))


def _load_dataset_with_fallback(
    repo_id: str, config: str | None = None, local_name: str | None = None
) -> Dataset:
    """Load dataset from local disk if available, otherwise from HuggingFace."""
    if local_name:
        local_path = LOCAL_DATASETS_DIR / local_name
        if local_path.exists():
            return load_from_disk(str(local_path))
    if config:
        return load_dataset(repo_id, config)
    return load_dataset(repo_id)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = _load_dataset_with_fallback(
        "openai/gsm8k", config="main", local_name="gsm8k"
    )
    data = data[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
            "dataset_type": "gsm8k",
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = _load_dataset_with_fallback("ankner/math-500", local_name="math-500")
    data = data[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
            "dataset_type": "math",
        }
    )
    return data


def kodcode_preprocess_test(test_str: str) -> str:
    """
    Minimal preprocessing:
      1) Remove the first 'from solution import ...' line
      2) Append calls to all `test_*` functions at the end so tests execute
    """

    lines = test_str.splitlines()

    # Remove first from solution import line
    new_lines = []
    removed = False
    for line in lines:
        if (not removed) and line.strip().startswith("from solution import"):
            removed = True
            continue
        new_lines.append(line)

    result = "\n".join(new_lines).rstrip()
    result += "\n"

    # Find all test function names
    test_funcs = re.findall(r"^def\s+(test_[A-Za-z0-9_]+)\s*\(", result, flags=re.M)

    # Append the calls
    if test_funcs:
        call_block = "\n" + "\n".join(f"{name}()" for name in test_funcs) + "\n"
        result += call_block

    return result


def get_kodcode_questions() -> Dataset:
    data = _load_dataset_with_fallback(
        "KodCode/KodCode-Light-RL-10K", local_name="kodcode"
    )
    if "train" in data:
        data = data["train"]

    def build(x):
        info = x["test_info"][0]
        fn_decl = info["function_declaration"]  # e.g., "def array_to_string(arr):"

        # User message (HumanEval-ish)
        user_msg = (
            f"You are an expert Python programmer, and here is your task:\n\n"
            f"{x['question']}\n\n"
            "Implement the function exactly as specified below."
        )

        # Generation prefix appended *after* applying the chat template
        gen_prefix = f"Here is the completed function:\n```python\n{fn_decl}\n"

        return {
            "prompt": [{"role": "user", "content": user_msg}],
            "gen_prefix": gen_prefix,
            "raw_prompt": fn_decl,
            "solution": x["solution"],
            "test": x["test"],
            "answer": kodcode_preprocess_test(x["test"]),
            "dataset_type": "kodcode",
            "function_name": info["function_name"],
            "function_declaration": fn_decl,
        }

    return data.map(build)


def get_gsm8k_and_math_questions(split="train", seed=42) -> Dataset:
    gsm8k_data = get_gsm8k_questions(split)
    math_data = get_math_questions(split)

    # Since size of GSM8k is approximately the same as Math (7.47K vs 7.5K),
    # it is probably safe to simply concatenate the two datasets to get the
    # 50:50 mixture.
    mixed_data = concatenate_datasets([gsm8k_data, math_data]).shuffle(seed=seed)

    return mixed_data


def get_gsm8k_and_math_and_kodcode_questions(split="train", seed=42) -> Dataset:
    gsm8k_data = get_gsm8k_questions(split)
    math_data = get_math_questions(split)
    kodcode_data = get_kodcode_questions()

    # GSM8k size: 7.47K, Math size: 7.5K, KodCode size: 10K
    mixed_data = concatenate_datasets([gsm8k_data, math_data, kodcode_data]).shuffle(
        seed=seed
    )

    return mixed_data
