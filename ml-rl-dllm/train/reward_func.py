#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
from typing import Callable

import torch

try:
    import evaluate as hf_evaluate
except ImportError:
    hf_evaluate = None

from common.parsing.parse_and_get_acc import check_gsm_correct
from common.parsing.parse_and_get_acc import extract_gsm_answer
from common.parsing.parse_and_get_acc import extract_gsm_answer_for_reward
from common.parsing.parser_utils import is_equiv
from common.parsing.parser_utils import last_boxed_only_string
from common.parsing.parser_utils import remove_boxed
from common.rewards import additive_proposal_reward
from common.rewards import apply_reward_quality_caps
from common.rewards import apply_negative_junk_penalty
from common.rewards import compute_budget_reward
from common.rewards import compute_reward_quality
from common.rewards import deliberation_bonus
from common.rewards import has_malformed_answer_structure
from common.rewards import patient_win_bonus_applies
from data.sanitize import sanitize_humaneval

code_eval = None


def _get_code_eval():
    global code_eval
    if code_eval is None:
        if hf_evaluate is None:
            raise ImportError(
                "The 'evaluate' package is required for code-eval rewards."
            )
        code_eval = hf_evaluate.load("code_eval")
    return code_eval


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def _parse_num(num_str: str) -> float:
    # Simple reformatter to handle cases like "10 000" and "999,999"
    reformatted_num_str = num_str.replace(" ", "").replace(",", "")
    return float(reformatted_num_str)


def _process_answers_gsm8k(parsed_responses, answer, pos_reward) -> list[float]:
    correctness_rewards = []
    for extracted, ground_truth in zip(parsed_responses, answer):
        correctness_rewards.append(
            pos_reward if check_gsm_correct(extracted, ground_truth) else 0.0
        )
    return correctness_rewards


def _multiplicative_step_scaling_reward_func(
    parsing_fn: Callable[[str], float | str | None],
    process_answers_fn: Callable[[list[float], list[float], float], list[float]],
    prompts,
    completions,
    answer,
    n_steps: torch.Tensor,
    L: int,
    step=None,
    run_name=None,
    pos_reward=1.0,
    alpha=1.0,
    **kwargs,
) -> list[float]:
    r"""Extracts answers from the completions using `parsing_fn`, then computes a reward
    of the form $\delta(\hat{y} = y) \cdot \left(\frac{L - steps + 1}{L}\right)^\alpha.
    Uses float comparison of extracted numbers instead of exact string matching.
    """

    responses = [completion[0]["content"] for completion in completions]

    if kwargs["dataset_type"][0] == "kodcode":
        parsed_responses = []
        for i, r in enumerate(responses):
            raw_prompt = kwargs["raw_prompt"][i]
            if not raw_prompt.endswith("\n"):
                raw_prompt = raw_prompt + "\n"
            parsed_responses.append(
                parsing_fn(raw_prompt + r, kwargs["function_name"][i])
            )

    else:
        parsed_responses = [parsing_fn(r) for r in responses]

    correctness_rewards = process_answers_fn(parsed_responses, answer, pos_reward)

    # Apply step-based scaling based on alpha:
    # - Alpha = 0.0: no scaling
    # - 0 < Alpha < 1: Gentle falloff (e.g., α=0.5 is square root)
    # - Alpha = 1: Linear decay
    # - Alpha > 1: Steep falloff (e.g., α=2 is quadratic)
    # - At step L: Goes to (1/L)^α, not exactly 0
    # NOTE: Clamps steps to max at L to make the math simpler.
    if alpha == 0.0:
        return correctness_rewards
    else:
        return [
            reward * ((L - min(steps.item(), L) + 1) / L) ** alpha
            for reward, steps in zip(correctness_rewards, n_steps)
        ]


def _additive_compute_reward_func(
    parsing_fn: Callable[[str], float | str | None],
    process_answers_fn: Callable[[list[float], list[float], float], list[float]],
    prompts,
    completions,
    answer,
    n_steps: torch.Tensor,
    L: int,
    step=None,
    run_name=None,
    pos_reward=1.0,
    alpha=1.0,
    **kwargs,
) -> list[float]:
    r"""Extracts answers from the completions using `parsing_fn`, then computes a reward
    of the form $\delta(\hat{y} = y) \cdot \left(\frac{L - steps + 1}{L}\right)^\alpha.
    Uses float comparison of extracted numbers instead of exact string matching.
    """
    responses = [completion[0]["content"] for completion in completions]

    if kwargs.get("dataset_type", [None])[0] == "kodcode":
        parsed_responses = []
        for i, r in enumerate(responses):
            raw_prompt = kwargs["raw_prompt"][i]
            if not raw_prompt.endswith("\n"):
                raw_prompt = raw_prompt + "\n"
            parsed_responses.append(
                parsing_fn(raw_prompt + r, kwargs["function_name"][i])
            )
    else:
        parsed_responses = [parsing_fn(r) for r in responses]

    correctness_rewards = process_answers_fn(parsed_responses, answer, pos_reward)

    if alpha == 0.0:
        return correctness_rewards
    else:
        return [
            reward - alpha * steps / L
            for reward, steps in zip(correctness_rewards, n_steps)
        ]


def xml_mult_reward(*args, **kwargs) -> list[float]:
    """
    Combines strict XML-format numeric answer matching with step-based reward scaling.
    """
    return _multiplicative_step_scaling_reward_func(
        extract_gsm_answer, _process_answers_gsm8k, *args, **kwargs
    )


def xml_add_reward(*args, **kwargs) -> list[float]:
    """
    Combines strict XML-format numeric answer matching with step-based reward scaling.
    """
    return _additive_compute_reward_func(
        extract_gsm_answer, _process_answers_gsm8k, *args, **kwargs
    )


def extract_answer_math(r) -> str | None:
    try:
        r = remove_boxed(last_boxed_only_string(r))
        return r
    except Exception:
        return None


def _process_answers_math(parsed_responses, answer, pos_reward) -> list[float]:
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    return [
        pos_reward if is_equiv(r, a) else 0.0 for r, a in zip(parsed_responses, answer)
    ]


def math_correctness_mult_reward(
    *args,
    **kwargs,
) -> list[float]:
    r"""Extracts answers from the completions using `parsing_fn`, then computes a reward
    of the form $\delta(\hat{y} = y) \cdot \left(\frac{L - steps + 1}{L}\right)^\alpha.
    Uses float comparison of extracted numbers instead of exact string matching.
    """
    return _multiplicative_step_scaling_reward_func(
        extract_answer_math, _process_answers_math, *args, **kwargs
    )


def math_correctness_add_reward(
    *args,
    **kwargs,
) -> list[float]:
    return _additive_compute_reward_func(
        extract_answer_math, _process_answers_math, *args, **kwargs
    )


def evaluate_code(generations, tests, pos_reward):
    """Evaluate code generations using HuggingFace's code_eval metric.

    :param generations: List of generation dicts from evaluate()
    :param tests: List of test cases
    :param pos_reward: Positive reward multiplier
    :return: List of pass@1 results
    """
    pass_at_1s = []
    for gen, test in zip(generations, tests):
        try:
            # # Since we have 1 generation per task, wrap each in a list
            predictions = [[gen]]
            # references = [[test] for test in tests]
            references = [test]

            # Compute pass@k with k=[1]
            # Returns tuple: (pass_at_k_dict, results_dict)
            pass_at_k, _ = _get_code_eval().compute(
                references=references, predictions=predictions, k=[1]
            )

            pass_at_1 = pass_at_k["pass@1"] * pos_reward

            pass_at_1s.append(pass_at_1)

        except Exception as e:
            print(f"Error during code evaluation: {e}")
            import traceback

            traceback.print_exc()
            pass_at_1s.append(0.0)

    return pass_at_1s


def kodcode_correctness_mult_reward(
    *args,
    **kwargs,
) -> list[float]:
    return _multiplicative_step_scaling_reward_func(
        sanitize_humaneval, evaluate_code, *args, **kwargs
    )


def mixed_correctness_mult_reward_func(
    prompts,
    completions,
    answer,
    n_steps,
    L,
    alpha,
    **kwargs,
) -> list[float]:
    dataset_type = kwargs["dataset_type"]

    # TODO: refactor to handle multi-dataset-batches
    # more elegantly... we are looping over the lists
    # inside each specific reward func anyway, so
    # could just flatten things instead of looping here, too
    if len(set(dataset_type)) > 1:
        rewards = []
        for i in range(len(prompts)):
            sample_kwargs = {
                k: [v[i]] if isinstance(v, list) else v for k, v in kwargs.items()
            }

            if dataset_type[i] == "gsm8k":
                r = xml_mult_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            elif dataset_type[i] == "math":
                r = math_correctness_mult_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            elif dataset_type[i] == "kodcode":
                r = kodcode_correctness_mult_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            else:
                raise ValueError(f"Dataset type {dataset_type[i]} not supported")
            rewards.append(r[0])
        return rewards
    else:
        # Uniform batch -> can just call the downstream func
        if dataset_type[0] == "gsm8k":
            return xml_mult_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        elif dataset_type[0] == "math":
            return math_correctness_mult_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        elif dataset_type[0] == "kodcode":
            return kodcode_correctness_mult_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        else:
            raise ValueError(f"Dataset type {dataset_type} not supported")


def kodcode_correctness_add_reward(
    *args,
    **kwargs,
) -> list[float]:
    return _additive_compute_reward_func(
        sanitize_humaneval, evaluate_code, *args, **kwargs
    )


def mixed_correctness_add_reward_func(
    prompts,
    completions,
    answer,
    n_steps,
    L,
    alpha,
    **kwargs,
) -> list[float]:
    dataset_type = kwargs["dataset_type"]

    # TODO: refactor to handle multi-dataset-batches
    # more elegantly... we are looping over the lists
    # inside each specific reward func anyway, so
    # could just flatten things instead of looping here, too
    if len(set(dataset_type)) > 1:
        rewards = []
        for i in range(len(prompts)):
            sample_kwargs = {
                k: [v[i]] if isinstance(v, list) else v for k, v in kwargs.items()
            }

            if dataset_type[i] == "gsm8k":
                r = xml_add_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            elif dataset_type[i] == "math":
                r = math_correctness_add_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            elif dataset_type[i] == "kodcode":
                r = kodcode_correctness_add_reward(
                    prompts=[prompts[i]],
                    completions=[completions[i]],
                    answer=[answer[i]],
                    n_steps=[n_steps[i]],
                    L=L,
                    alpha=alpha,
                    **sample_kwargs,
                )
            else:
                raise ValueError(f"Dataset type {dataset_type[i]} not supported")
            rewards.append(r[0])
        return rewards
    else:
        if dataset_type[0] == "gsm8k":
            return xml_add_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        elif dataset_type[0] == "math":
            return math_correctness_add_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        elif dataset_type[0] == "kodcode":
            return kodcode_correctness_add_reward(
                prompts=prompts,
                completions=completions,
                answer=answer,
                n_steps=n_steps,
                L=L,
                alpha=alpha,
                **kwargs,
            )
        else:
            raise ValueError(f"Dataset type {dataset_type} not supported")


def _completion_to_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def _value_for_index(value, index, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _mixed_correctness_scores(
    prompts,
    completions,
    answer,
    pos_reward,
    **kwargs,
) -> list[float]:
    dataset_type = kwargs["dataset_type"]
    responses = [_completion_to_text(completion) for completion in completions]
    scores = []
    for i, dataset_name in enumerate(dataset_type):
        if dataset_name == "gsm8k":
            extractor_mode = kwargs.get("reward_gsm_extractor", "robust")
            parsed = [
                extract_gsm_answer_for_reward(
                    responses[i],
                    mode=extractor_mode,
                )
            ]
            scores.extend(_process_answers_gsm8k(parsed, [answer[i]], pos_reward))
        elif dataset_name == "math":
            parsed = [extract_answer_math(responses[i])]
            scores.extend(_process_answers_math(parsed, [answer[i]], pos_reward))
        elif dataset_name == "kodcode":
            raw_prompt = kwargs["raw_prompt"][i]
            if not raw_prompt.endswith("\n"):
                raw_prompt = raw_prompt + "\n"
            parsed = sanitize_humaneval(raw_prompt + responses[i], kwargs["function_name"][i])
            scores.extend(evaluate_code([parsed], [answer[i]], pos_reward))
        else:
            raise ValueError(f"Dataset type {dataset_name} not supported")
    return scores


def _quality_kwargs(kwargs: dict, index: int | None = None) -> dict:
    format_ok = kwargs.get("format_ok")
    if index is not None:
        format_ok = _value_for_index(format_ok, index, None)
    return {
        "extractor_mode": kwargs.get("reward_gsm_extractor", "robust"),
        "reward_quality_mode": kwargs.get("reward_quality_mode", "none"),
        "format_ok": format_ok,
        "answer_span_partial_credit": kwargs.get("answer_span_partial_credit", 0.4),
        "malformed_answer_factor": kwargs.get("malformed_answer_factor", 0.5),
        "source_answer_marker_credit": kwargs.get(
            "source_answer_marker_credit",
            0.20,
        ),
        "source_final_line_credit": kwargs.get("source_final_line_credit", 0.15),
        "source_final_window_credit": kwargs.get("source_final_window_credit", 0.08),
        "source_incomplete_answer_tag_credit": kwargs.get(
            "source_incomplete_answer_tag_credit",
            0.03,
        ),
        "source_other_answer_span_credit": kwargs.get(
            "source_other_answer_span_credit",
            0.05,
        ),
        "enable_repetition_penalty": kwargs.get("enable_repetition_penalty", False),
        "repetition_min_tokens": kwargs.get("repetition_min_tokens", 8),
        "repetition_number_run_threshold": kwargs.get(
            "repetition_number_run_threshold",
            3,
        ),
        "repetition_token_run_threshold": kwargs.get(
            "repetition_token_run_threshold",
            4,
        ),
        "repetition_bigram_run_threshold": kwargs.get(
            "repetition_bigram_run_threshold",
            3,
        ),
        "repetition_answer_span_number_threshold": kwargs.get(
            "repetition_answer_span_number_threshold",
            2,
        ),
        "repetition_unique_ratio_threshold": kwargs.get(
            "repetition_unique_ratio_threshold",
            0.25,
        ),
        "repetition_max_freq_threshold": kwargs.get(
            "repetition_max_freq_threshold",
            0.25,
        ),
        "repetition_penalty_factor": kwargs.get("repetition_penalty_factor", 0.5),
        "zero_reward_if_repeated_answer_span": kwargs.get(
            "zero_reward_if_repeated_answer_span",
            True,
        ),
    }


def _quality_cap_for_task_score(
    *,
    task_score: float,
    quality,
    max_reward_if_malformed: float | None = None,
    max_reward_if_repetitive: float | None = None,
):
    is_clean_format = bool(quality.strict_correct) or bool(quality.format_ok)
    repeated_junk_detected = bool(quality.repeated_junk_detected) or (
        float(quality.repetition_penalty) < 1.0
    )
    return apply_reward_quality_caps(
        reward=task_score,
        is_malformed=not is_clean_format,
        repeated_junk_detected=repeated_junk_detected,
        max_reward_if_malformed=max_reward_if_malformed,
        max_reward_if_repetitive=max_reward_if_repetitive,
    )


def _mixed_task_score_details(
    prompts,
    completions,
    answer,
    pos_reward,
    apply_quality_caps: bool = True,
    scale_task_score_by_pos_reward: bool = True,
    **kwargs,
) -> list[dict]:
    dataset_type = kwargs["dataset_type"]
    responses = [_completion_to_text(completion) for completion in completions]
    details = []
    for i, dataset_name in enumerate(dataset_type):
        if dataset_name == "gsm8k":
            quality = compute_reward_quality(
                responses[i],
                answer[i],
                **_quality_kwargs(kwargs, i),
            )
            quality_task_score = quality.task_score
            if scale_task_score_by_pos_reward:
                raw_task_score = quality_task_score * float(pos_reward)
            else:
                raw_task_score = quality_task_score
            if apply_quality_caps:
                cap = _quality_cap_for_task_score(
                    task_score=raw_task_score,
                    quality=quality,
                    max_reward_if_malformed=kwargs.get("max_reward_if_malformed"),
                    max_reward_if_repetitive=kwargs.get("max_reward_if_repetitive"),
                )
                task_score_after_caps = cap.reward_after_quality_caps
                reward_before_quality_caps = cap.reward_before_quality_caps
                malformed_cap_applied = cap.malformed_reward_cap_applied
                repetitive_cap_applied = cap.repetitive_reward_cap_applied
            else:
                task_score_after_caps = raw_task_score
                reward_before_quality_caps = raw_task_score
                malformed_cap_applied = False
                repetitive_cap_applied = False
            details.append(
                {
                    "quality_task_score_before_scaling": quality_task_score,
                    "quality_task_score_after_caps": task_score_after_caps,
                    "completion_length_normalizer": 1.0,
                    "raw_task_score": raw_task_score,
                    "task_score": task_score_after_caps,
                    "parsed_answer": quality.parsed_answer,
                    "answer_source": quality.answer_source,
                    "answer_span_source": quality.answer_span_source,
                    "strict_correct": quality.strict_correct,
                    "answer_span_correct": quality.answer_span_correct,
                    "robust_correct": quality.robust_correct,
                    "format_ok": quality.format_ok,
                    "repetition_penalty": quality.repetition_penalty,
                    "repeated_char_run_detected": (
                        quality.repeated_char_run_detected
                    ),
                    "repeated_number_run_detected": (
                        quality.repeated_number_run_detected
                    ),
                    "repeated_token_run_detected": (
                        quality.repeated_token_run_detected
                    ),
                    "repeated_ngram_detected": quality.repeated_ngram_detected,
                    "repeated_answer_span_detected": (
                        quality.repeated_answer_span_detected
                    ),
                    "low_unique_ratio_detected": quality.low_unique_ratio_detected,
                    "high_token_freq_detected": quality.high_token_freq_detected,
                    "repeated_junk_detected": quality.repeated_junk_detected,
                    "has_malformed_answer_structure": (
                        has_malformed_answer_structure(responses[i])
                    ),
                    "reward_before_quality_caps": (
                        reward_before_quality_caps
                    ),
                    "reward_after_quality_caps": task_score_after_caps,
                    "malformed_reward_cap_applied": (
                        malformed_cap_applied
                    ),
                    "repetitive_reward_cap_applied": (
                        repetitive_cap_applied
                    ),
                    "quality_cap_applicable": True,
                }
            )
        elif dataset_name == "math":
            parsed = [extract_answer_math(responses[i])]
            score = _process_answers_math(parsed, [answer[i]], pos_reward)[0]
            details.append(
                {
                    "task_score": score,
                    "parsed_answer": parsed[0],
                    "quality_cap_applicable": False,
                }
            )
        elif dataset_name == "kodcode":
            raw_prompt = kwargs["raw_prompt"][i]
            if not raw_prompt.endswith("\n"):
                raw_prompt = raw_prompt + "\n"
            parsed = sanitize_humaneval(
                raw_prompt + responses[i],
                kwargs["function_name"][i],
            )
            score = evaluate_code([parsed], [answer[i]], pos_reward)[0]
            details.append(
                {
                    "task_score": score,
                    "parsed_answer": parsed,
                    "quality_cap_applicable": False,
                }
            )
        else:
            raise ValueError(f"Dataset type {dataset_name} not supported")
    return details


def mixed_additive_proposal_reward_func(
    prompts,
    completions,
    answer,
    n_steps,
    L,
    alpha=0.01,
    pos_reward=1.0,
    **kwargs,
) -> list[float]:
    details = _mixed_task_score_details(
        prompts=prompts,
        completions=completions,
        answer=answer,
        pos_reward=pos_reward,
        **kwargs,
    )
    lambda_compute = kwargs.get("reward_lambda_compute", alpha)
    normalize_compute = kwargs.get("reward_normalize_compute", True)
    return [
        additive_proposal_reward(
            task_score=detail["task_score"],
            nfe_used=_value_for_index(n_steps, i, 0),
            max_steps=L,
            lambda_compute=lambda_compute,
            normalize_compute=normalize_compute,
        )
        for i, detail in enumerate(details)
    ]


def mixed_multiplicative_budget_reward_func(
    prompts,
    completions,
    answer,
    n_steps,
    L,
    alpha=None,
    pos_reward=1.0,
    **kwargs,
) -> list[float]:
    beta = kwargs.get("reward_beta", 2.0)
    mu = kwargs.get("reward_mu", 0.5)
    rho = kwargs.get("reward_rho", 0.1)
    reward_budget_mode = kwargs.get("reward_budget_mode", "cap")
    if kwargs.get("reward_type") == "task_only":
        reward_budget_mode = "none"
    task_only_reward = reward_budget_mode == "none"

    details = _mixed_task_score_details(
        prompts=prompts,
        completions=completions,
        answer=answer,
        pos_reward=1.0,
        apply_quality_caps=not task_only_reward,
        scale_task_score_by_pos_reward=False,
        **kwargs,
    )
    threshold_reward_hard_zero = kwargs.get("threshold_reward_hard_zero", False)
    enable_deliberation_probe = kwargs.get("enable_deliberation_probe", False)
    deliberation_bonus_weight = kwargs.get("deliberation_bonus_weight", 0.0)

    rewards = []
    final_task_scores = []
    threshold_gates = []
    for i, detail in enumerate(details):
        if task_only_reward:
            assert float(detail.get("completion_length_normalizer", 1.0)) == 1.0
        task_score = float(detail["task_score"])
        if not task_only_reward:
            task_score *= float(pos_reward)
        final_task_scores.append(task_score)
        budget = compute_budget_reward(
            task_score=task_score,
            nfe_used=_value_for_index(n_steps, i, 0),
            target_budget=_value_for_index(kwargs.get("target_budget"), i, L),
            max_steps=L,
            remask_count=_value_for_index(kwargs.get("remask_count"), i, 0),
            reward_budget_mode=reward_budget_mode,
            reward_beta=beta,
            reward_mu=mu,
            reward_rho=rho,
            threshold_reward_hard_zero=threshold_reward_hard_zero,
        )
        reward = budget.reward
        threshold_gates.append(budget.threshold_gate)
        if enable_deliberation_probe and not task_only_reward:
            early_text = _value_for_index(kwargs.get("early_generated_text"), i, None)
            early_score = 0.0
            if early_text is not None and kwargs["dataset_type"][i] == "gsm8k":
                early_quality_kwargs = _quality_kwargs(kwargs, i)
                early_quality_kwargs["format_ok"] = None
                early_quality = compute_reward_quality(
                    early_text,
                    answer[i],
                    **early_quality_kwargs,
                )
                early_score = early_quality.task_score * float(pos_reward)
            _, bonus = deliberation_bonus(
                early_task_score=early_score,
                final_task_score=task_score,
                threshold_gate=budget.threshold_gate,
                deliberation_bonus_weight=deliberation_bonus_weight,
            )
            reward += bonus
        rewards.append(float(reward))

    if not task_only_reward:
        styles = [
            _value_for_index(kwargs.get("rollout_compute_style"), i, "normal")
            for i in range(len(rewards))
        ]
        num_generations = int(kwargs.get("num_generations", 0) or 0)
        patient_win_bonus = float(kwargs.get("patient_win_bonus", 0.0) or 0.0)
        patient_win_margin = float(kwargs.get("patient_win_margin", 0.05) or 0.05)
        if patient_win_bonus > 0.0 and num_generations > 0:
            for start in range(0, len(rewards), num_generations):
                end = min(start + num_generations, len(rewards))
                group_styles = styles[start:end]
                if "normal" not in group_styles or "patient" not in group_styles:
                    continue
                normal_idx = start + group_styles.index("normal")
                patient_idx = start + group_styles.index("patient")
                target_budget = _value_for_index(
                    kwargs.get("target_budget"),
                    patient_idx,
                    L,
                )
                if patient_win_bonus_applies(
                    patient_task_score=final_task_scores[patient_idx],
                    normal_task_score=final_task_scores[normal_idx],
                    patient_nfe=_value_for_index(n_steps, patient_idx, 0),
                    target_budget=target_budget,
                    patient_win_margin=patient_win_margin,
                ):
                    rewards[patient_idx] += (
                        patient_win_bonus * threshold_gates[patient_idx]
                    )
        max_reward_if_malformed = kwargs.get("max_reward_if_malformed")
        max_reward_if_repetitive = kwargs.get("max_reward_if_repetitive")
        for i, detail in enumerate(details):
            if not detail.get("quality_cap_applicable", False):
                continue
            is_clean_format = bool(detail.get("strict_correct")) or bool(
                detail.get("format_ok")
            )
            repeated_junk_detected = bool(detail.get("repeated_junk_detected")) or (
                float(detail.get("repetition_penalty", 1.0)) < 1.0
            )
            cap = apply_reward_quality_caps(
                reward=rewards[i],
                is_malformed=not is_clean_format,
                repeated_junk_detected=repeated_junk_detected,
                max_reward_if_malformed=max_reward_if_malformed,
                max_reward_if_repetitive=max_reward_if_repetitive,
            )
            rewards[i] = cap.reward_after_quality_caps
    if kwargs.get("enable_negative_junk_penalty", False):
        for i, detail in enumerate(details):
            if not detail.get("quality_cap_applicable", False):
                continue
            repeated_junk_detected = bool(detail.get("repeated_junk_detected")) or (
                float(detail.get("repetition_penalty", 1.0)) < 1.0
            )
            negative = apply_negative_junk_penalty(
                reward=rewards[i],
                strict_correct=bool(detail.get("strict_correct")),
                repeated_junk_detected=repeated_junk_detected,
                malformed_junk_detected=bool(
                    detail.get("has_malformed_answer_structure")
                ),
                answer_source=detail.get("answer_source"),
                enable_negative_junk_penalty=True,
                repeated_junk_negative_reward=kwargs.get(
                    "repeated_junk_negative_reward",
                    0.02,
                ),
                malformed_junk_negative_reward=kwargs.get(
                    "malformed_junk_negative_reward",
                    0.01,
                ),
                incomplete_answer_tag_negative_reward=kwargs.get(
                    "incomplete_answer_tag_negative_reward",
                    0.01,
                ),
                min_reward_floor=kwargs.get("min_reward_floor", -0.05),
            )
            rewards[i] = negative.reward_after_negative_junk_penalty
    return rewards


def mixed_correctness_reward_func(
    *args,
    **kwargs,
) -> list[float]:
    """Pure correctness reward for mixed datasets without step penalty."""
    kwargs = {**kwargs, "alpha": 0.0}
    return mixed_correctness_add_reward_func(*args, **kwargs)
