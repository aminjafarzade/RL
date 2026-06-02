#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import json
import re

import tiktoken

from common.parsing.parser_utils import is_equiv
from common.parsing.parser_utils import last_boxed_only_string
from common.parsing.parser_utils import remove_boxed


def count_effective_tokens(text):
    """Count tokens in generated text.

    :param text: Text to tokenize
    :return: Number of tokens (excluding endoftext markers)
    """
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


# ============================================================================
# Answer Extraction Functions (dataset-specific logic)
# ============================================================================


def extract_gsm_answer(raw_generation: str) -> float | None:
    """Extract numeric answer from GSM8K generation.

    :param raw_generation: Generated text to parse
    :return: Extracted numeric answer, or None if no valid answer found
    """
    parsed_answer = None

    # Try \boxed{} format first
    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if (
                boxed_content
                and boxed_content != "..."
                and not re.match(r"^\.+$", boxed_content)
            ):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass

    # Try <answer></answer> format
    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass
    return parsed_answer


def extract_math_answer(raw_generation: str) -> str | None:
    """Extract LaTeX answer from MATH generation.

    :param raw_generation: Generated text to parse
    :return: Extracted LaTeX answer string, or None if no valid answer found
    """
    parsed_answer = None

    # Try \boxed{} format first
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except Exception:
        pass

    # Try <answer></answer> format
    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()

    return parsed_answer


def extract_code_answer(item: dict) -> float:
    """Extract pass@1 from code generation (HumanEval/MBPP).

    :param item: Item dictionary containing code evaluation results
    :return: Pass@1 score (0.0 to 1.0)
    """
    return item.get("pass@1", 0.0)


# ============================================================================
# Answer Checking Functions (dataset-specific correctness logic)
# ============================================================================


def check_gsm_correct(extracted, ground_truth) -> bool:
    """Check if GSM8K answer is correct.

    :param extracted: Extracted numeric answer
    :param ground_truth: Ground truth numeric answer
    :return: True if extracted matches ground truth
    """
    return extracted is not None and extracted == ground_truth


def check_math_correct(extracted, ground_truth) -> bool:
    """Check if MATH answer is correct using equivalence.

    :param extracted: Extracted LaTeX answer string
    :param ground_truth: Ground truth LaTeX answer string
    :return: True if extracted is mathematically equivalent to ground truth
    """
    if extracted is None:
        return False
    return is_equiv(extracted, ground_truth)


def check_code_correct(extracted, ground_truth) -> bool:
    """Check if code passed tests (pass@1 is 1.0).

    :param extracted: Pass@1 score from code evaluation
    :param ground_truth: Ground truth (not used - checks pass@1 directly)
    :return: True if pass@1 equals 1.0
    """
    return extracted == 1.0


# ============================================================================
# Generic Parser (unified logic for all datasets)
# ============================================================================


def parse_answers_generic(
    json_path=None,
    json_data=None,
    extract_fn=None,
    check_fn=None,
    item_key="generations",
):
    """Generic parser for all datasets.

    :param json_path: Path to JSON file (if loading from file)
    :param json_data: JSON data dict (if already loaded)
    :param extract_fn: Function to extract answer from generation
    :param check_fn: Function to check if answer is correct
    :param item_key: Key for generation text in item dict
    :return: Tuple of (total_correct, total_processed, processed_items, total_effective_tokens, steps, wall_times)
    """
    # Load data
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []
    steps = []
    wall_times = []

    for item in data.get("generations", []):
        total_processed += 1

        # Get generation and ground truth
        raw_generation = item.get(item_key, "")
        ground_truth = item.get("ground_truth")
        steps.append(item.get("steps", 0))
        wall_times.append(item.get("wall_time", 0.0))

        # Count tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract answer
        extracted_answer = extract_fn(item) if extract_fn else None

        # Check correctness
        is_correct = check_fn(extracted_answer, ground_truth) if check_fn else False
        if is_correct:
            total_correct += 1

        # Store processed item
        processed_items.append(
            {
                "question": item.get("question", item.get("text", "")),
                "raw_generation": raw_generation,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
        steps,
        wall_times,
    )


# ============================================================================
# Dataset-specific wrappers (maintain same API as original)
# ============================================================================


def parse_gsm_answers(json_path=None, json_data=None):
    """Parse GSM8K answers.

    :param json_path: Path to JSON file containing GSM8K generations
    :param json_data: Pre-loaded JSON data dict
    :return: Tuple of (total_correct, total_processed, processed_items, total_effective_tokens, steps, wall_times)
    """
    return parse_answers_generic(
        json_path=json_path,
        json_data=json_data,
        extract_fn=lambda item: extract_gsm_answer(item.get("generations", "")),
        check_fn=check_gsm_correct,
        item_key="generations",
    )


def parse_math_answers(json_path=None, json_data=None):
    """Parse MATH answers.

    :param json_path: Path to JSON file containing MATH generations
    :param json_data: Pre-loaded JSON data dict
    :return: Tuple of (total_correct, total_processed, processed_items, total_effective_tokens, steps, wall_times)
    """
    return parse_answers_generic(
        json_path=json_path,
        json_data=json_data,
        extract_fn=lambda item: extract_math_answer(item.get("generations", "")),
        check_fn=check_math_correct,
        item_key="generations",
    )


def parse_code_answers(json_path=None, json_data=None):
    """Parse code answers (HumanEval/MBPP).

    :param json_path: Path to JSON file containing code generations
    :param json_data: Pre-loaded JSON data dict
    :return: Tuple of (total_correct, total_processed, processed_items, total_effective_tokens, steps, wall_times)
    """
    return parse_answers_generic(
        json_path=json_path,
        json_data=json_data,
        extract_fn=extract_code_answer,
        check_fn=check_code_correct,
        item_key="generation_sanitized",
    )
