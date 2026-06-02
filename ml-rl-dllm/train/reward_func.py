#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
from typing import Callable

import evaluate as hf_evaluate
import torch

from common.parsing.parse_and_get_acc import extract_gsm_answer
from common.parsing.parser_utils import is_equiv
from common.parsing.parser_utils import last_boxed_only_string
from common.parsing.parser_utils import remove_boxed
from common.rewards import additive_proposal_reward
from common.rewards import multiplicative_budget_reward
from data.sanitize import sanitize_humaneval

code_eval = hf_evaluate.load("code_eval")


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
        if extracted is None:
            correctness_rewards.append(0.0)
            continue

        try:
            # Eval script doesn't cast GT to float, just assumes it is;
            # for some reason it seems we get the GT as a string here?
            ground_truth_float = _parse_num(ground_truth)
        except (ValueError, TypeError):
            raise ValueError(f"Ground truth '{ground_truth}' is not a valid number")

        try:
            # Handle both string and float return types from parsing_fn
            extracted_float = (
                _parse_num(extracted) if isinstance(extracted, str) else extracted
            )
            is_correct = abs(extracted_float - ground_truth_float) < 1e-6
            correctness_rewards.append(pos_reward if is_correct else 0.0)
        except (ValueError, TypeError):
            correctness_rewards.append(0.0)
            continue
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
            pass_at_k, _ = code_eval.compute(
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
            parsed = [extract_gsm_answer(responses[i])]
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
    scores = _mixed_correctness_scores(
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
            task_score=score,
            nfe_used=_value_for_index(n_steps, i, 0),
            max_steps=L,
            lambda_compute=lambda_compute,
            normalize_compute=normalize_compute,
        )
        for i, score in enumerate(scores)
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
    scores = _mixed_correctness_scores(
        prompts=prompts,
        completions=completions,
        answer=answer,
        pos_reward=pos_reward,
        **kwargs,
    )
    beta = kwargs.get("reward_beta", 2.0)
    mu = kwargs.get("reward_mu", 0.5)
    rho = kwargs.get("reward_rho", 0.1)
    return [
        pos_reward
        * multiplicative_budget_reward(
            exact_match=score > 0.0,
            nfe_used=_value_for_index(n_steps, i, 0),
            target_budget=_value_for_index(kwargs.get("target_budget"), i, L),
            max_steps=L,
            remask_count=_value_for_index(kwargs.get("remask_count"), i, 0),
            beta=beta,
            mu=mu,
            rho=rho,
        )
        for i, score in enumerate(scores)
    ]


def mixed_correctness_reward_func(
    *args,
    **kwargs,
) -> list[float]:
    """Pure correctness reward for mixed datasets without step penalty."""
    kwargs = {**kwargs, "alpha": 0.0}
    return mixed_correctness_add_reward_func(*args, **kwargs)
