from common.remasking.span_finder import AnswerSpan
from common.remasking.strategies import RemaskState
from common.remasking.strategies import VerifierAnswerRemask
from common.verifiers.base import VerifierResult


def _bad_verifier_result():
    return VerifierResult(
        parse_ok=True,
        format_ok=True,
        arithmetic_ok=False,
        final_answer="42",
        final_answer_span_text="42",
        verifier_score=0.2,
        flags={},
        error=None,
    )


def test_verifier_answer_remask_respects_max_remasks_per_sample():
    strategy = VerifierAnswerRemask(max_remasks_per_sample=1, remask_cooldown_steps=0)
    state = RemaskState()
    span = AnswerSpan(token_indices=[3], text="42")

    first = strategy.select(_bad_verifier_result(), span, state, current_step=1)
    assert first.should_remask
    strategy.record(state, first, step=1)

    second = strategy.select(
        _bad_verifier_result(),
        AnswerSpan(token_indices=[4], text="43"),
        state,
        current_step=2,
    )
    assert not second.should_remask
    assert second.reason == "max_remasks_reached"


def test_verifier_answer_remask_respects_cooldown():
    strategy = VerifierAnswerRemask(
        max_remasks_per_sample=2,
        remask_cooldown_steps=2,
    )
    state = RemaskState()
    first = strategy.select(
        _bad_verifier_result(),
        AnswerSpan(token_indices=[3], text="42"),
        state,
        current_step=3,
    )
    strategy.record(state, first, step=3)

    second = strategy.select(
        _bad_verifier_result(),
        AnswerSpan(token_indices=[4], text="43"),
        state,
        current_step=4,
    )

    assert not second.should_remask
    assert second.reason == "cooldown"


def test_verifier_answer_remask_does_not_remask_prompt_tokens():
    strategy = VerifierAnswerRemask(max_remasks_per_sample=1)
    state = RemaskState()
    decision = strategy.select(
        _bad_verifier_result(),
        AnswerSpan(token_indices=[0, 1], text="42"),
        state,
        current_step=5,
        prompt_token_count=2,
    )
    assert not decision.should_remask
    assert decision.reason == "prompt_span_rejected"


def test_verifier_answer_remask_rejects_repeated_span():
    strategy = VerifierAnswerRemask(
        max_remasks_per_sample=2,
        remask_cooldown_steps=0,
    )
    state = RemaskState()
    span = AnswerSpan(token_indices=[3], text="42")
    first = strategy.select(_bad_verifier_result(), span, state, current_step=1)
    strategy.record(state, first, step=1)

    second = strategy.select(_bad_verifier_result(), span, state, current_step=2)
    assert not second.should_remask
    assert second.reason == "repeat_span_rejected"
