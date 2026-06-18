#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


@dataclass
class VerifierResult:
    parse_ok: bool
    format_ok: bool
    arithmetic_ok: bool | None
    final_answer: str | None
    final_answer_span_text: str | None
    verifier_score: float
    flags: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseVerifier:
    def verify(self, text: str, prompt: str | None = None) -> VerifierResult:
        raise NotImplementedError
