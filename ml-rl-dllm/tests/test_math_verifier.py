import math

from common.verifiers.math_verifier import MathVerifier
from common.verifiers.math_verifier import check_arithmetic
from common.verifiers.math_verifier import normalize_numeric_answer
from common.verifiers.math_verifier import safe_eval_arithmetic


def test_math_verifier_parses_supported_answer_formats():
    verifier = MathVerifier()

    assert verifier.verify("work\n#### 123").final_answer == "123"
    assert verifier.verify("final is \\boxed{456}").final_answer == "456"
    assert verifier.verify("<answer>$1,234</answer>").final_answer == "1234"
    assert verifier.verify("The answer is 7.5.").final_answer == "7.5"
    assert verifier.verify("reasoning 10 then final 42").final_answer == "42"


def test_math_verifier_normalizes_numeric_forms():
    assert normalize_numeric_answer("$1,234") == "1234"
    assert normalize_numeric_answer("10.50") == "10.5"
    assert normalize_numeric_answer("1/2") == "0.5"
    assert normalize_numeric_answer("\\frac{3}{4}") == "0.75"


def test_arithmetic_checker_valid_invalid_and_absent_equations():
    assert check_arithmetic("48 + 24 = 72") is True
    assert check_arithmetic("12 * 4 = 48 and 6 × 4 = 24") is True
    assert check_arithmetic("300 - 12 = 288") is True
    assert check_arithmetic("6 ÷ 4 = 2") is False
    assert check_arithmetic("No explicit equation here.") is None


def test_arithmetic_checker_rejects_unsafe_or_invalid_expressions():
    assert check_arithmetic("2 / 0 = 0") is False
    try:
        safe_eval_arithmetic("__import__('os').system('echo bad')")
    except Exception:
        pass
    else:
        raise AssertionError("unsafe arithmetic expression should not evaluate")


def test_verifier_score_is_label_free_and_finite():
    result = MathVerifier().verify("48 + 24 = 72, so \\boxed{72}")
    assert result.parse_ok is True
    assert result.format_ok is True
    assert result.arithmetic_ok is True
    assert math.isfinite(result.verifier_score)
    assert 0.0 <= result.verifier_score <= 1.0
