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


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    span_text: str
    normalized: str | None
    source: str


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


def check_arithmetic(text: str) -> bool | None:
    found_equation = False
    for match in _EQUATION_RE.finditer(text or ""):
        found_equation = True
        lhs = match.group("lhs").strip()
        rhs = match.group("rhs").strip()
        try:
            lhs_value = safe_eval_arithmetic(lhs)
            rhs_value = safe_eval_arithmetic(rhs)
        except Exception:
            return False
        if lhs_value != rhs_value:
            return False
    if not found_equation:
        return None
    return True


class MathVerifier(BaseVerifier):
    def verify(self, text: str, prompt: str | None = None) -> VerifierResult:
        try:
            candidate = extract_final_answer(text)
            parse_ok = candidate is not None and candidate.normalized is not None
            format_ok = candidate is not None and candidate.source != "fallback_number"
            arithmetic_ok = check_arithmetic(text)

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
