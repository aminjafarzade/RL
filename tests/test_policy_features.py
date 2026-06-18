from types import SimpleNamespace

import torch

from common.policy_features import build_policy_extra_features
from common.policy_features import get_policy_extra_feature_names


def test_budget_feature_builder_returns_expected_shape():
    names = get_policy_extra_feature_names(
        SimpleNamespace(enable_budget_conditioning=True, enable_verifier=False)
    )
    mask_index = torch.tensor([[True, False, True]])
    confidence = torch.tensor([[0.1, 0.8, 0.4]])
    entropy = torch.tensor([[1.0, 0.2, 0.5]])
    steps_taken = torch.tensor([4])

    features = build_policy_extra_features(
        feature_names=names,
        mask_index=mask_index,
        confidence=confidence,
        entropy=entropy,
        steps_taken=steps_taken,
        max_steps=16,
        target_budget=8,
    )

    assert features.shape == (1, 3, len(names))


def test_verifier_feature_builder_encodes_label_free_verifier_state():
    names = get_policy_extra_feature_names(
        SimpleNamespace(enable_budget_conditioning=False, enable_verifier=True)
    )
    mask_index = torch.tensor([[True, False]])
    confidence = torch.tensor([[0.2, 0.7]])
    entropy = torch.tensor([[1.1, 0.3]])
    steps_taken = torch.tensor([2])
    verifier_states = [
        {
            "parse_ok": True,
            "format_ok": False,
            "arithmetic_ok": None,
            "verifier_score": 0.6,
            "final_answer_span_text": "42",
        }
    ]

    features = build_policy_extra_features(
        feature_names=names,
        mask_index=mask_index,
        confidence=confidence,
        entropy=entropy,
        steps_taken=steps_taken,
        max_steps=8,
        verifier_states=verifier_states,
        answer_span_indicator=torch.tensor([[0.0, 1.0]]),
    )

    assert features.shape == (1, 2, len(names))
    assert features[0, 0, names.index("parse_ok")] == 1.0
    assert features[0, 0, names.index("arithmetic_unknown")] == 1.0
    assert features[0, 1, names.index("answer_span_indicator")] == 1.0
