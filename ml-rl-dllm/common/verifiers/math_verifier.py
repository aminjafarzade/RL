#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import ast
import operator
import re
from dataclasses import dataclass
from decimal import Decimal
from decimal import InvalidOperation
from decimal import getcontext
from fractions import Fraction
from collections import Counter
from typing import Callable

from common.verifiers.base import BaseVerifier
from common.verifiers.base import VerifierResult

getcontext().prec = 32

_LATEX_FRAC_RE = re.compile(
    r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:)\s*([^\n\r<]+)", re.IGNORECASE
)
_NUMBER_RE = re.compile(
    r"[-+]?\s*\$?\s*(?:\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
    r"(?:\d[\d,]*|\.\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?)"
)
_EQUATION_RE = re.compile(
    r"(?P<lhs>[-+\s$\\{}.,/\d()]+(?:[+*xX×÷/\-][-+\s$\\{}.,/\d()]+)+)"
    r"\s*=\s*"
    r"(?P<rhs>[-+\s$\\{}.,/\d()]+)"
)
_ANSWER_MARKER_RE = re.compile(
    r"####|\\boxed\s*\{|<answer>|the\s+answer\s+is|answer\s*:",
    re.IGNORECASE,
)
_SIMPLE_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*|\.\d+)(?:\.\d+)?")
_BINARY_ARITHMETIC_OP_RE = re.compile(
    r"\d\s*(?:[+*xX×÷/]|-)\s*[-+]?\s*(?:\d|\$|[({])"
)

DEFAULT_MAX_ARITHMETIC_TEXT_CHARS = 2048
DEFAULT_MAX_EQUATION_MATCHES = 16
DEFAULT_REPEATED_NUMBER_SKIP_THRESHOLD = 8
_EQUATION_CONTEXT_CHARS = 96
_ARITHMETIC_ALLOWED_CHARS = set("0123456789+-*/xX×÷()., ${}\\\t\n\r")


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    span_text: str
    normalized: str | None
    source: str


@dataclass(frozen=True)
class ArithmeticCheckResult:
    arithmetic_ok: bool | None
    flags: dict[str, object]


def has_answer_marker(text: str) -> bool:
    return bool(_ANSWER_MARKER_RE.search(text or ""))


def _replace_latex_fracs(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", current)
    return current


def _strip_boxed(text: str) -> str:
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1)
    return text


def _to_fraction(text: str) -> Fraction | None:
    if text is None:
        return None
    normalized = _replace_latex_fracs(_strip_boxed(str(text)))
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\$", "$")
    normalized = normalized.strip()

    # Keep only a single simple numeric expression for final-answer parsing.
    normalized = re.sub(r"^[a-zA-Z]\s*=\s*", "", normalized)
    normalized = normalized.replace("$", "").replace(",", "").replace(" ", "")
    normalized = normalized.strip("{}[]")
    normalized = normalized.rstrip(".;:")
    if not normalized:
        return None

    if "/" in normalized and not any(op in normalized for op in ["+", "*"]):
        parts = [part.strip("{}[]()") for part in normalized.split("/")]
        if len(parts) == 2:
            try:
                numerator = Decimal(parts[0])
                denominator = Decimal(parts[1])
                if denominator == 0:
                    return None
                return Fraction(numerator) / Fraction(denominator)
            except (InvalidOperation, ValueError, ZeroDivisionError):
                return None

    try:
        return Fraction(Decimal(normalized.strip("()")))
    except (InvalidOperation, ValueError):
        return None


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal_value.normalize(), "f")


def normalize_numeric_answer(text: str | None) -> str | None:
    value = _to_fraction(text)
    if value is None:
        return None
    return _format_fraction(value)


def _candidate_from_text(text: str, source: str) -> AnswerCandidate:
    stripped = text.strip()
    if source in {"hash", "answer_is"}:
        number_match = _NUMBER_RE.search(stripped)
        if number_match:
            stripped = number_match.group(0).strip()
    boxed = _BOXED_RE.search(stripped)
    if boxed:
        stripped = boxed.group(1).strip()
    return AnswerCandidate(
        text=text.strip(),
        span_text=stripped,
        normalized=normalize_numeric_answer(stripped),
        source=source,
    )


def extract_final_answer(text: str) -> AnswerCandidate | None:
    if not text:
        return None

    candidates: list[AnswerCandidate] = []
    for match in _HASH_ANSWER_RE.finditer(text):
        candidates.append(_candidate_from_text(match.group(1), "hash"))
    for match in _BOXED_RE.finditer(text):
        candidates.append(_candidate_from_text(match.group(1), "boxed"))
    for match in _ANSWER_TAG_RE.finditer(text):
        candidates.append(_candidate_from_text(match.group(1), "answer_tag"))
    for match in _ANSWER_IS_RE.finditer(text):
        candidates.append(_candidate_from_text(match.group(1), "answer_is"))

    for candidate in candidates:
        if candidate.normalized is not None:
            return candidate

    fallback_matches = list(_NUMBER_RE.finditer(text))
    for match in reversed(fallback_matches):
        candidate = _candidate_from_text(match.group(0), "fallback_number")
        if candidate.normalized is not None:
            return candidate
    return candidates[-1] if candidates else None


_SAFE_BIN_OPS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_SAFE_UNARY_OPS: dict[type[ast.unaryop], Callable[[Fraction], Fraction]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _normalize_arithmetic_expr(expr: str) -> str:
    expr = _replace_latex_fracs(expr)
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("x", "*").replace("X", "*")
    expr = expr.replace("$", "").replace(",", "")
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.strip().strip(";:")
    if expr.endswith(".") and expr.count(".") == 1 and not re.search(r"\d\.$", expr):
        expr = expr[:-1]
    return expr


def _eval_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return _SAFE_BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"Unsafe arithmetic expression node: {type(node).__name__}")


def safe_eval_arithmetic(expr: str) -> Fraction:
    normalized = _normalize_arithmetic_expr(expr)
    parsed = ast.parse(normalized, mode="eval")
    return _eval_ast(parsed)


def _is_arithmetic_char(char: str) -> bool:
    return char in _ARITHMETIC_ALLOWED_CHARS


def _trim_lhs_expression(text: str, equal_index: int) -> str:
    start = max(0, equal_index - _EQUATION_CONTEXT_CHARS)
    snippet = text[start:equal_index]
    pos = len(snippet) - 1
    while pos >= 0 and _is_arithmetic_char(snippet[pos]):
        pos -= 1
    return snippet[pos + 1 :].strip()


def _trim_rhs_expression(text: str, equal_index: int) -> str:
    end = min(len(text), equal_index + 1 + _EQUATION_CONTEXT_CHARS)
    snippet = text[equal_index + 1 : end]
    pos = 0
    while pos < len(snippet) and _is_arithmetic_char(snippet[pos]):
        pos += 1
    return snippet[:pos].strip()


def _looks_like_equation_side(lhs: str, rhs: str) -> bool:
    return (
        bool(lhs)
        and bool(rhs)
        and bool(re.search(r"\d", lhs))
        and bool(re.search(r"\d", rhs))
        and bool(_BINARY_ARITHMETIC_OP_RE.search(lhs))
    )


def _simple_number_key(text: str) -> str:
    return text.replace(",", "").replace(" ", "").lstrip("+")


def _repeated_number_to_skip(
    text: str,
    threshold: int,
) -> tuple[str | None, int]:
    if threshold <= 0:
        return None, 0
    counts: Counter[str] = Counter()
    for match in _SIMPLE_NUMBER_RE.finditer(text):
        key = _simple_number_key(match.group(0))
        if not key:
            continue
        counts[key] += 1
        if counts[key] >= threshold:
            return key, counts[key]
    return None, 0


def _iter_equation_candidates(text: str):
    start = 0
    while True:
        equal_index = text.find("=", start)
        if equal_index < 0:
            return
        lhs = _trim_lhs_expression(text, equal_index)
        rhs = _trim_rhs_expression(text, equal_index)
        if _looks_like_equation_side(lhs, rhs):
            yield lhs, rhs
        start = equal_index + 1


def _check_arithmetic_with_metadata(
    text: str,
    max_arithmetic_text_chars: int = DEFAULT_MAX_ARITHMETIC_TEXT_CHARS,
    max_equation_matches: int = DEFAULT_MAX_EQUATION_MATCHES,
    repeated_number_skip_threshold: int = DEFAULT_REPEATED_NUMBER_SKIP_THRESHOLD,
) -> ArithmeticCheckResult:
    source_text = text or ""
    limit = max(0, int(max_arithmetic_text_chars))
    arithmetic_text = source_text[:limit] if limit else ""
    flags: dict[str, object] = {
        "arithmetic_text_truncated": len(source_text) > len(arithmetic_text),
        "arithmetic_text_chars": len(arithmetic_text),
        "arithmetic_checked_equations": 0,
        "arithmetic_skipped_repeated_number": False,
        "arithmetic_repeated_number": None,
        "arithmetic_repeated_number_count": 0,
        "arithmetic_match_limit_hit": False,
    }

    repeated_number, repeated_count = _repeated_number_to_skip(
        arithmetic_text,
        int(repeated_number_skip_threshold),
    )
    if repeated_number is not None:
        flags.update(
            {
                "arithmetic_skipped_repeated_number": True,
                "arithmetic_repeated_number": repeated_number,
                "arithmetic_repeated_number_count": repeated_count,
            }
        )
        return ArithmeticCheckResult(arithmetic_ok=None, flags=flags)

    found_equation = False
    checked = 0
    seen: set[tuple[str, str]] = set()
    for lhs, rhs in _iter_equation_candidates(arithmetic_text):
        key = (lhs, rhs)
        if key in seen:
            continue
        seen.add(key)
        if checked >= int(max_equation_matches):
            flags["arithmetic_match_limit_hit"] = True
            break
        found_equation = True
        checked += 1
        flags["arithmetic_checked_equations"] = checked
        try:
            lhs_value = safe_eval_arithmetic(lhs)
            rhs_value = safe_eval_arithmetic(rhs)
        except Exception:
            return ArithmeticCheckResult(arithmetic_ok=False, flags=flags)
        if lhs_value != rhs_value:
            return ArithmeticCheckResult(arithmetic_ok=False, flags=flags)
    if not found_equation:
        return ArithmeticCheckResult(arithmetic_ok=None, flags=flags)
    return ArithmeticCheckResult(arithmetic_ok=True, flags=flags)


def check_arithmetic(
    text: str,
    max_arithmetic_text_chars: int = DEFAULT_MAX_ARITHMETIC_TEXT_CHARS,
    max_equation_matches: int = DEFAULT_MAX_EQUATION_MATCHES,
    repeated_number_skip_threshold: int = DEFAULT_REPEATED_NUMBER_SKIP_THRESHOLD,
) -> bool | None:
    return _check_arithmetic_with_metadata(
        text,
        max_arithmetic_text_chars=max_arithmetic_text_chars,
        max_equation_matches=max_equation_matches,
        repeated_number_skip_threshold=repeated_number_skip_threshold,
    ).arithmetic_ok


class MathVerifier(BaseVerifier):
    def __init__(
        self,
        max_arithmetic_text_chars: int = DEFAULT_MAX_ARITHMETIC_TEXT_CHARS,
        max_equation_matches: int = DEFAULT_MAX_EQUATION_MATCHES,
        repeated_number_skip_threshold: int = DEFAULT_REPEATED_NUMBER_SKIP_THRESHOLD,
    ):
        self.max_arithmetic_text_chars = max_arithmetic_text_chars
        self.max_equation_matches = max_equation_matches
        self.repeated_number_skip_threshold = repeated_number_skip_threshold

    def verify(self, text: str, prompt: str | None = None) -> VerifierResult:
        try:
            candidate = extract_final_answer(text)
            parse_ok = candidate is not None and candidate.normalized is not None
            format_ok = candidate is not None and candidate.source != "fallback_number"
            arithmetic_check = _check_arithmetic_with_metadata(
                text,
                max_arithmetic_text_chars=self.max_arithmetic_text_chars,
                max_equation_matches=self.max_equation_matches,
                repeated_number_skip_threshold=self.repeated_number_skip_threshold,
            )
            arithmetic_ok = arithmetic_check.arithmetic_ok

            score = 0.0
            score += 0.4 if parse_ok else 0.0
            score += 0.2 if format_ok else 0.0
            if arithmetic_ok is True:
                score += 0.4
            elif arithmetic_ok is None:
                score += 0.2

            flags = {
                "answer_source": candidate.source if candidate else None,
                "has_answer_marker": has_answer_marker(text),
                "prompt_provided": prompt is not None,
                **arithmetic_check.flags,
            }
            return VerifierResult(
                parse_ok=parse_ok,
                format_ok=format_ok,
                arithmetic_ok=arithmetic_ok,
                final_answer=candidate.normalized if candidate else None,
                final_answer_span_text=candidate.span_text if candidate else None,
                verifier_score=float(max(0.0, min(1.0, score))),
                flags=flags,
                error=None,
            )
        except Exception as exc:
            return VerifierResult(
                parse_ok=False,
                format_ok=False,
                arithmetic_ok=None,
                final_answer=None,
                final_answer_span_text=None,
                verifier_score=0.0,
                flags={"prompt_provided": prompt is not None},
                error=str(exc),
            )
