import pytest

from common.parsing.parse_and_get_acc import extract_gsm_answer_for_reward
from common.parsing.parse_and_get_acc import extract_gsm_answer_with_source
from common.parsing.parse_and_get_acc import gsm_correctness_score
from common.parsing.parse_and_get_acc import numeric_equal
from common.parsing.parse_and_get_acc import parse_gsm_answers


def test_strict_and_robust_gsm_extraction():
    assert extract_gsm_answer_for_reward(r"\boxed{100}", mode="strict") == "100"
    assert extract_gsm_answer_for_reward(r"\boxed{100}", mode="answer_span") == "100"
    assert extract_gsm_answer_for_reward(r"\boxed{100}", mode="robust") == "100"
    assert extract_gsm_answer_for_reward("<answer>100</answer>", mode="strict") == "100"
    assert (
        extract_gsm_answer_for_reward("<answer>100</answer>", mode="answer_span")
        == "100"
    )
    assert extract_gsm_answer_for_reward("<answer>100</answer>", mode="robust") == "100"

    assert extract_gsm_answer_for_reward("The answer is 100.", mode="strict") is None
    assert (
        extract_gsm_answer_for_reward("The answer is 100.", mode="answer_span")
        == "100"
    )
    assert extract_gsm_answer_for_reward("The answer is 100.", mode="robust") == "100"
    assert extract_gsm_answer_for_reward("So 48 + 24 = 72", mode="robust") == "72"


def test_gsm_numeric_equal():
    assert numeric_equal("100", 100)
    assert numeric_equal("100.0", "100")
    assert numeric_equal("1,000", "1000")
    assert numeric_equal("$1,000", "1000")
    assert not numeric_equal("101", "100")


def test_unknown_gsm_reward_extractor_mode_is_rejected():
    with pytest.raises(ValueError):
        extract_gsm_answer_for_reward("The answer is 100.", mode="unknown")


def test_robust_extractor_can_reward_fallback_answer():
    text = "The answer is 100."
    answer = "100"

    strict_score = gsm_correctness_score(text, answer, extractor_mode="strict")
    robust_score = gsm_correctness_score(text, answer, extractor_mode="robust")

    assert strict_score == 0.0
    assert robust_score > 0.0


def test_answer_span_extractor_reports_strict_sources():
    assert (
        extract_gsm_answer_with_source("<answer>350</answer>", mode="answer_span")
        .source
        == "strict_answer_tag"
    )
    assert (
        extract_gsm_answer_with_source(r"\boxed{74}", mode="answer_span").source
        == "strict_boxed"
    )


def test_answer_span_extractor_accepts_final_answer_marker():
    result = extract_gsm_answer_with_source(
        "Reasoning goes here.\nTherefore the answer is 85.",
        mode="answer_span",
    )

    assert result.answer == "85"
    assert result.source == "answer_marker"


def test_answer_span_extractor_accepts_number_before_closing_answer_marker():
    result = extract_gsm_answer_with_source(
        "some reasoning 21 minutes.\n</answer>",
        mode="answer_span",
    )

    assert result.answer == "21"
    assert result.source == "incomplete_answer_tag"


def test_empty_answer_tag_blocks_answer_span_fallback_but_not_robust():
    text = "reasoning has 1 ... <answer>\n</answer>"

    assert extract_gsm_answer_for_reward(text, mode="answer_span") is None
    assert extract_gsm_answer_for_reward(text, mode="robust") == "1"


def test_answer_span_extractor_rejects_reasoning_only_number():
    text = "12 + 3 = 15. lots of text no final answer"

    assert extract_gsm_answer_for_reward(text, mode="answer_span") is None
    assert extract_gsm_answer_for_reward(text, mode="robust") == "15"


def test_answer_span_extractor_rejects_repetitive_junk_without_marker():
    text = " ".join(["hulaula"] * 40 + ["21"] + ["hulaula"] * 40)

    assert extract_gsm_answer_for_reward(text, mode="answer_span") is None
    assert extract_gsm_answer_for_reward(text, mode="robust") == "21"


def test_parse_gsm_answers_strict_mode_requires_strict_format():
    total_correct, total_processed, *_ = parse_gsm_answers(
        json_data={
            "generations": [
                {
                    "generations": "The answer is 100.",
                    "ground_truth": "100",
                }
            ]
        },
        extractor_mode="strict",
    )

    assert total_processed == 1
    assert total_correct == 0


def test_parse_gsm_answers_robust_mode_counts_fallback_answer():
    total_correct, total_processed, *_ = parse_gsm_answers(
        json_data={
            "generations": [
                {
                    "generations": "The answer is 100.",
                    "ground_truth": "100",
                }
            ]
        },
        extractor_mode="robust",
    )

    assert total_processed == 1
    assert total_correct == 1
