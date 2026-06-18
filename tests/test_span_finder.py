from common.remasking.span_finder import find_final_answer_span
from common.verifiers.base import VerifierResult


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    all_special_ids = [0, 99]

    def __init__(self):
        self.vocab = {
            1: "The",
            2: " answer",
            3: " is",
            4: " 42",
            5: ".",
            6: " because",
            7: " 6",
        }

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            pieces.append(self.vocab[token_id])
        return "".join(pieces)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("force prefix-decode fallback")


def _verifier_result(span_text="42"):
    return VerifierResult(
        parse_ok=True,
        format_ok=True,
        arithmetic_ok=None,
        final_answer=span_text,
        final_answer_span_text=span_text,
        verifier_score=0.8,
        flags={},
        error=None,
    )


def test_span_finder_maps_final_answer_to_token_span():
    tokenizer = FakeTokenizer()
    token_ids = [1, 2, 3, 4, 5]
    text = tokenizer.decode(token_ids)

    span = find_final_answer_span(
        tokenizer=tokenizer,
        decoded_text=text,
        generated_token_ids=token_ids,
        verifier_result=_verifier_result("42"),
    )

    assert span is not None
    assert span.token_indices == [3]
    assert span.text == "42"


def test_span_finder_falls_back_to_small_numeric_suffix():
    tokenizer = FakeTokenizer()
    token_ids = [1, 2, 3, 7, 5]
    text = tokenizer.decode(token_ids)

    span = find_final_answer_span(
        tokenizer=tokenizer,
        decoded_text=text,
        generated_token_ids=token_ids,
        verifier_result=_verifier_result("123"),
    )

    assert span is not None
    assert span.token_indices == [3]


def test_span_finder_does_not_return_padding_only_span():
    tokenizer = FakeTokenizer()
    span = find_final_answer_span(
        tokenizer=tokenizer,
        decoded_text="",
        generated_token_ids=[0, 99],
        verifier_result=_verifier_result("42"),
    )
    assert span is None
