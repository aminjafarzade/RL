#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Sequence

import torch

from common.verifiers.base import VerifierResult


@dataclass(frozen=True)
class AnswerSpan:
    token_indices: list[int]
    text: str
    char_start: int | None = None
    char_end: int | None = None
    mean_confidence: float | None = None
    min_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(token_ids: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    return [int(x) for x in token_ids]


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _special_token_ids(tokenizer, explicit: set[int] | None = None) -> set[int]:
    ids = set(explicit or set())
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    for value in getattr(tokenizer, "all_special_ids", []) or []:
        if value is not None:
            ids.add(int(value))
    return ids


def _find_char_span(decoded_text: str, span_text: str | None) -> tuple[int, int] | None:
    if not decoded_text or not span_text:
        return None
    start = decoded_text.rfind(span_text)
    if start < 0:
        compact_span = span_text.replace(",", "").replace("$", "").strip()
        compact_text = decoded_text.replace(",", "").replace("$", "")
        compact_start = compact_text.rfind(compact_span)
        if compact_start < 0:
            return None
        # Use the compact index only as a fallback; it is conservative because
        # remasking will map by decoded prefixes and may return no span.
        start = compact_start
        span_text = compact_span
    return start, start + len(span_text)


def _map_with_offsets(tokenizer, text: str, token_ids: list[int], start: int, end: int):
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except Exception:
        return None

    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    encoded_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if not offsets or encoded_ids is None or len(offsets) != len(token_ids):
        return None
    if [int(x) for x in encoded_ids] != token_ids:
        return None
    indices = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end
    ]
    return indices or None


def _map_with_prefix_decode(tokenizer, token_ids: list[int], start: int, end: int):
    indices = []
    previous_len = 0
    for idx in range(len(token_ids)):
        current_len = len(_decode(tokenizer, token_ids[: idx + 1]))
        if current_len > start and previous_len < end:
            indices.append(idx)
        previous_len = current_len
    return indices or None


def _fallback_numeric_suffix(
    tokenizer,
    token_ids: list[int],
    min_token_index: int,
    disallowed_ids: set[int],
) -> list[int] | None:
    for idx in range(len(token_ids) - 1, min_token_index - 1, -1):
        if token_ids[idx] in disallowed_ids:
            continue
        token_text = _decode(tokenizer, [token_ids[idx]])
        if any(ch.isdigit() for ch in token_text):
            return [idx]
    for idx in range(len(token_ids) - 1, min_token_index - 1, -1):
        if token_ids[idx] not in disallowed_ids:
            return [idx]
    return None


def _confidence_stats(
    token_indices: list[int],
    confidences: Sequence[float] | torch.Tensor | None,
) -> tuple[float | None, float | None]:
    if confidences is None or not token_indices:
        return None, None
    if isinstance(confidences, torch.Tensor):
        values = confidences.detach().cpu()
        selected = [float(values[idx].item()) for idx in token_indices]
    else:
        selected = [float(confidences[idx]) for idx in token_indices]
    return sum(selected) / len(selected), min(selected)


def find_final_answer_span(
    tokenizer,
    decoded_text: str,
    generated_token_ids: Sequence[int] | torch.Tensor,
    verifier_result: VerifierResult,
    confidences: Sequence[float] | torch.Tensor | None = None,
    min_token_index: int = 0,
    special_token_ids: set[int] | None = None,
    allow_special_tokens: bool = False,
    remask_span_mode: str = "numeric_only",
) -> AnswerSpan | None:
    token_ids = _as_list(generated_token_ids)
    if not token_ids:
        return None

    disallowed_ids = set()
    if not allow_special_tokens:
        disallowed_ids = _special_token_ids(tokenizer, special_token_ids)

    span_text = verifier_result.final_answer_span_text or verifier_result.final_answer
    if remask_span_mode == "numeric_only" and span_text is not None:
        if not any(ch.isdigit() for ch in span_text):
            return None

    char_span = _find_char_span(decoded_text, span_text)
    token_indices = None
    char_start = char_end = None
    if char_span is not None:
        char_start, char_end = char_span
        token_indices = _map_with_offsets(
            tokenizer, decoded_text, token_ids, char_start, char_end
        )
        if token_indices is None:
            token_indices = _map_with_prefix_decode(
                tokenizer, token_ids, char_start, char_end
            )

    if token_indices is None:
        token_indices = _fallback_numeric_suffix(
            tokenizer, token_ids, min_token_index, disallowed_ids
        )

    if not token_indices:
        return None
    token_indices = [
        idx
        for idx in token_indices
        if idx >= min_token_index
        and (allow_special_tokens or token_ids[idx] not in disallowed_ids)
    ]
    if not token_indices:
        return None

    mean_conf, min_conf = _confidence_stats(token_indices, confidences)
    text = span_text or _decode(tokenizer, [token_ids[idx] for idx in token_indices])
    return AnswerSpan(
        token_indices=token_indices,
        text=text,
        char_start=char_start,
        char_end=char_end,
        mean_confidence=mean_conf,
        min_confidence=min_conf,
    )
