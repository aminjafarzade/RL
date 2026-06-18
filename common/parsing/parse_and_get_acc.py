#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import json
import re
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:
    tiktoken = None

from common.parsing.parser_utils import is_equiv
from common.parsing.parser_utils import last_boxed_only_string
from common.parsing.parser_utils import remove_boxed
from common.verifiers.math_verifier import extract_final_answer
from common.verifiers.math_verifier import (
    normalize_numeric_answer as verifier_normalize_numeric_answer,
)

_GSM_NUMBER_RE = re.compile(
    r"[-+]?\s*\$?\s*(?:\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
    r"(?:\d[\d,]*|\.\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)"
)
_COMPLETE_ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_MARKER_RE = re.compile(
    r"(?:answer\s+is|final\s+answer)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GSMAnswerExtraction:
    answer: str | None
    source: str = "none"


def count_effective_tokens(text):
    """Count tokens in generated text.

    :param text: Text to tokenize
    :return: Number of tokens (excluding endoftext markers)
    """
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    if tiktoken is None:
        return len(re.findall(r"\S+", text))
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


# ============================================================================
# Answer Extraction Functions (dataset-specific logic)
# ============================================================================


def normalize_gsm_numeric_answer(value) -> str | None:
    """Normalize GSM8K numeric answers for reward/eval comparison."""
    if value is None:
        return None
    normalized = verifier_normalize_numeric_answer(str(value))
    if normalized is not None:
        return normalized

    text = str(value).strip().replace(",", "").replace("$", "")
    text = text.rstrip(".;:")
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return str(number)


def _normalize_strict_gsm_content(text: str) -> str | None:
    normalized = normalize_gsm_numeric_answer(text)
    if normalized is not None:
        return normalized
    candidate = extract_final_answer(text or "")
    if candidate is None:
        return None
    if candidate.normalized is not None:
        return normalize_gsm_numeric_answer(candidate.normalized)
    return normalize_gsm_numeric_answer(candidate.span_text or candidate.text)


def _numeric_candidates(text: str) -> list[tuple[str, int, int]]:
    candidates = []
    for match in _GSM_NUMBER_RE.finditer(text or ""):
        normalized = normalize_gsm_numeric_answer(match.group(0))
        if normalized is not None:
            candidates.append((normalized, match.start(), match.end()))
    return candidates


def _number_from_region(text: str, prefer: str = "last") -> str | None:
    candidates = _numeric_candidates(text)
    if not candidates:
        return None
    if prefer == "first":
        return candidates[0][0]
    return candidates[-1][0]


def _has_empty_complete_answer_tag(text: str) -> bool:
    return any(not match.group(1).strip() for match in _COMPLETE_ANSWER_TAG_RE.finditer(text or ""))


def _last_non_empty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_final_line_number_like(line: str) -> bool:
    candidates = _numeric_candidates(line)
    if len(candidates) != 1:
        return False
    if "=" in line:
        return False
    text_without_number = (
        line[: candidates[0][1]] + line[candidates[0][2] :]
    ).strip()
    text_without_number = text_without_number.strip(".:,;!?()[]{}<>/\\")
    if not text_without_number:
        return True
    allowed_unit_words = {
        "dollars",
        "dollar",
        "hours",
        "hour",
        "minutes",
        "minute",
        "meters",
        "meter",
        "miles",
        "mile",
        "years",
        "year",
        "days",
        "day",
        "cents",
        "cent",
    }
    words = re.findall(r"[A-Za-z]+", text_without_number)
    return bool(words) and all(word.lower() in allowed_unit_words for word in words)


def _has_repetitive_junk(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text or "")
    if len(words) < 30:
        return False
    lowered = [word.lower() for word in words]
    unique_count = len(set(lowered))
    top_count = max(lowered.count(word) for word in set(lowered))
    return top_count / len(lowered) >= 0.25 or unique_count / len(lowered) <= 0.25


def _last_non_negated_marker(text: str) -> re.Match | None:
    last_match = None
    for match in _ANSWER_MARKER_RE.finditer(text or ""):
        prefix = text[max(0, match.start() - 8) : match.start()].lower()
        if re.search(r"\bno\s+$", prefix):
            continue
        last_match = match
    return last_match


def _extract_gsm_answer_strict_with_source(raw_generation: str) -> GSMAnswerExtraction:
    if not raw_generation:
        return GSMAnswerExtraction(None, "none")

    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if (
                boxed_content
                and boxed_content != "..."
                and not re.match(r"^\.+$", boxed_content)
            ):
                parsed_answer = _normalize_strict_gsm_content(boxed_content)
                if parsed_answer is not None:
                    return GSMAnswerExtraction(parsed_answer, "strict_boxed")

    for answer_match in _COMPLETE_ANSWER_TAG_RE.finditer(raw_generation):
        answer_text = answer_match.group(1).strip()
        if answer_text:
            parsed_answer = _normalize_strict_gsm_content(answer_text)
            if parsed_answer is not None:
                return GSMAnswerExtraction(parsed_answer, "strict_answer_tag")
    return GSMAnswerExtraction(None, "none")


def _extract_gsm_answer_answer_span_with_source(
    raw_generation: str,
) -> GSMAnswerExtraction:
    strict = _extract_gsm_answer_strict_with_source(raw_generation)
    if strict.answer is not None:
        return strict
    text = raw_generation or ""
    if not text.strip():
        return GSMAnswerExtraction(None, "none")
    if _has_empty_complete_answer_tag(text):
        return GSMAnswerExtraction(None, "none")

    marker = _last_non_negated_marker(text)
    if marker is not None:
        answer = _number_from_region(text[marker.end() :], prefer="first")
        if answer is not None:
            return GSMAnswerExtraction(answer, "answer_marker")

    lower_text = text.lower()
    last_open = lower_text.rfind("<answer>")
    last_close = lower_text.rfind("</answer>")
    if last_open >= 0 and last_open > last_close:
        answer = _number_from_region(text[last_open + len("<answer>") :], prefer="last")
        if answer is not None:
            return GSMAnswerExtraction(answer, "incomplete_answer_tag")

    if last_close >= 0:
        before_close = text[max(0, last_close - 80) : last_close]
        answer = _number_from_region(before_close, prefer="last")
        if answer is not None:
            return GSMAnswerExtraction(answer, "incomplete_answer_tag")

    final_line = _last_non_empty_line(text)
    final_line_number_like = _is_final_line_number_like(final_line)
    if final_line_number_like:
        answer = _number_from_region(final_line, prefer="last")
        if answer is not None:
            return GSMAnswerExtraction(answer, "final_line")

    if _has_repetitive_junk(text):
        return GSMAnswerExtraction(None, "none")

    if final_line:
        line_has_marker = _last_non_negated_marker(final_line) is not None
        if line_has_marker:
            answer = _number_from_region(final_line, prefer="last")
            if answer is not None:
                return GSMAnswerExtraction(answer, "final_line")

    final_window = text[-80:]
    window_candidates = _numeric_candidates(final_window)
    if len(window_candidates) == 1 and "=" not in final_window:
        return GSMAnswerExtraction(window_candidates[0][0], "final_window")

    return GSMAnswerExtraction(None, "none")


def _extract_gsm_answer_robust_with_source(raw_generation: str) -> GSMAnswerExtraction:
    strict = _extract_gsm_answer_strict_with_source(raw_generation)
    if strict.answer is not None:
        return strict

    candidate = extract_final_answer(raw_generation or "")
    if candidate is None:
        return GSMAnswerExtraction(None, "none")
    if getattr(candidate, "normalized", None) is not None:
        return GSMAnswerExtraction(
            normalize_gsm_numeric_answer(candidate.normalized),
            "robust_anywhere",
        )
    for attr in ("span_text", "text"):
        value = getattr(candidate, attr, None)
        normalized = normalize_gsm_numeric_answer(value)
        if normalized is not None:
            return GSMAnswerExtraction(normalized, "robust_anywhere")
    return GSMAnswerExtraction(
        normalize_gsm_numeric_answer(candidate),
        "robust_anywhere",
    )


def extract_gsm_answer_with_source(
    raw_generation: str,
    mode: str = "answer_span",
) -> GSMAnswerExtraction:
    if mode == "strict":
        return _extract_gsm_answer_strict_with_source(raw_generation)
    if mode == "answer_span":
        return _extract_gsm_answer_answer_span_with_source(raw_generation)
    if mode == "robust":
        return _extract_gsm_answer_robust_with_source(raw_generation)
    raise ValueError(
        f"Unknown reward_gsm_extractor={mode!r}; expected 'strict', "
        "'answer_span', or 'robust'"
    )


def extract_gsm_answer(raw_generation: str) -> str | None:
    """Extract strict-format numeric answer from GSM8K generation.

    This preserves the historical reward format requirement: only boxed answers
    and complete <answer>...</answer> tags are accepted. Numeric normalization is
    shared with the verifier so comma/decimal/string forms compare consistently.

    :param raw_generation: Generated text to parse
    :return: Extracted numeric answer, or None if no valid answer found
    """
    return extract_gsm_answer_with_source(raw_generation, mode="strict").answer


def extract_gsm_answer_strict(raw_generation: str) -> str | None:
    """Strict GSM8K extraction used for ablations and format diagnostics."""
    return extract_gsm_answer(raw_generation)


def extract_gsm_answer_robust(raw_generation: str) -> str | None:
    """GSM8K extraction that falls back to verifier-style final number parsing."""
    return extract_gsm_answer_with_source(raw_generation, mode="robust").answer


def extract_gsm_answer_answer_span(raw_generation: str) -> str | None:
    """GSM8K extraction from strict or final-answer-like answer spans."""
    return extract_gsm_answer_with_source(raw_generation, mode="answer_span").answer


def extract_gsm_answer_for_reward(
    raw_generation: str,
    mode: str = "robust",
) -> str | None:
    """Select strict, answer-span, or robust GSM8K reward extraction."""
    return extract_gsm_answer_with_source(raw_generation, mode=mode).answer


def numeric_equal(pred, gold, tol: float = 1e-6) -> bool:
    """Numeric-safe equality for GSM8K answers."""
    pred_norm = normalize_gsm_numeric_answer(pred)
    gold_norm = normalize_gsm_numeric_answer(gold)
    if pred_norm is None or gold_norm is None:
        return False
    try:
        return abs(float(pred_norm) - float(gold_norm)) <= tol
    except (TypeError, ValueError):
        return str(pred_norm).strip() == str(gold_norm).strip()


def gsm_correctness_score(
    raw_generation: str,
    ground_truth,
    extractor_mode: str = "robust",
    pos_reward: float = 1.0,
) -> float:
    parsed = extract_gsm_answer_for_reward(raw_generation, mode=extractor_mode)
    return float(pos_reward) if numeric_equal(parsed, ground_truth) else 0.0


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
    return numeric_equal(extracted, ground_truth)


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


def parse_gsm_answers(json_path=None, json_data=None, extractor_mode: str = "strict"):
    """Parse GSM8K answers.

    :param json_path: Path to JSON file containing GSM8K generations
    :param json_data: Pre-loaded JSON data dict
    :param extractor_mode: GSM extractor mode: strict or robust
    :return: Tuple of (total_correct, total_processed, processed_items, total_effective_tokens, steps, wall_times)
    """
    return parse_answers_generic(
        json_path=json_path,
        json_data=json_data,
        extract_fn=lambda item: extract_gsm_answer_for_reward(
            item.get("generations", ""),
            mode=extractor_mode,
        ),
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
