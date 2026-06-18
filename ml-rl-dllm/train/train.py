#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

from common.cuda_env import set_cuda_visible_devices_from_argv


set_cuda_visible_devices_from_argv()

import accelerate
import torch
import transformers
import trl
import wandb
from dotenv import load_dotenv
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import ModelConfig
from trl import TrlParser

import train.reward_func as reward_func
from common.config import Config
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from common.policy_features import get_policy_extra_feature_names
from common.rewards import select_reward_functions_from_config
from common.rewards import validate_reward_type_matches_functions
from common.s3 import S3UploadCallback
from common.s3 import download_s3_checkpoint
from common.s3 import get_latest_s3_checkpoint
from data.data_utils import get_gsm8k_and_math_and_kodcode_questions
from data.data_utils import get_gsm8k_and_math_questions
from data.data_utils import get_gsm8k_questions
from data.data_utils import get_kodcode_questions
from data.data_utils import get_math_questions
from data.data_utils import set_random_seed
from train.trainer import Trainer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

torch.set_float32_matmul_precision("high")

print("=== Library Versions ===")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"trl: {trl.__version__}")
try:
    import flash_attn

    print(f"flash_attn: {flash_attn.__version__}")
except ImportError:
    print("flash_attn: not installed")
print("========================")


def get_reward_functions(config: Config):
    """Get reward functions based on config."""
    if config.reward_functions is not None:
        reward_functions = []
        for func_name in config.reward_functions:
            func = getattr(reward_func, func_name, None)
            if func is None:
                raise ValueError(
                    f"Unknown reward function: {func_name}. Function not found in reward_func module."
                )
            reward_functions.append(func)
        return reward_functions

    selection = select_reward_functions_from_config(config)
    reward_functions = []
    for func_name in selection.reward_function_names:
        func = getattr(reward_func, func_name, None)
        if func is None:
            raise ValueError(
                f"Unknown reward function: {func_name}. Function not found in reward_func module."
            )
        reward_functions.append(func)
    return reward_functions


load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    print("HF_TOKEN not found; continuing without an explicit Hugging Face token.")


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}


def _resolve_policy_checkpoint_file(checkpoint_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(checkpoint_path)))
    if not expanded.exists():
        raise FileNotFoundError(
            f"policy_checkpoint_path does not exist: {checkpoint_path}"
        )
    if expanded.is_file():
        return expanded

    preferred_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "policy.safetensors",
        "policy.pt",
        "policy.bin",
        "checkpoint.pt",
    )
    for name in preferred_names:
        candidate = expanded / name
        if candidate.exists():
            return candidate

    candidates = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        candidates.extend(sorted(expanded.glob(pattern)))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidate_names = ", ".join(path.name for path in candidates[:8])
        raise FileNotFoundError(
            "policy_checkpoint_path contains multiple candidate policy files "
            f"without a preferred name: {checkpoint_path}. Candidates: {candidate_names}"
        )
    raise FileNotFoundError(
        "policy_checkpoint_path must point to a policy checkpoint file or a "
        "directory containing model.safetensors or pytorch_model.bin: "
        f"{checkpoint_path}"
    )


def _load_policy_checkpoint(policy: PolicyHFWrapper, checkpoint_path: str, device):
    checkpoint_file = _resolve_policy_checkpoint_file(checkpoint_path)
    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a .safetensors policy checkpoint requires safetensors."
            ) from exc
        state_dict = load_file(str(checkpoint_file), device="cpu")
    else:
        state_dict = torch.load(
            checkpoint_file,
            map_location="cpu",
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Policy checkpoint did not contain a state dict: {checkpoint_file}"
        )

    load_result = policy.load_state_dict(state_dict, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "Policy checkpoint keys did not match the configured policy. "
            f"missing={missing}, unexpected={unexpected}, path={checkpoint_file}"
        )
    policy.to(device)
    print(
        "Loaded lightweight policy checkpoint "
        f"from {checkpoint_file} "
        f"(missing={missing}, unexpected={unexpected})"
    )
    return {
        "policy_checkpoint_path_effective": str(checkpoint_file),
        "policy_checkpoint_loaded": True,
        "policy_checkpoint_missing_keys": missing,
        "policy_checkpoint_unexpected_keys": unexpected,
    }


def _validate_mixed_reward_functions(grpo_config):
    if grpo_config.reward_functions is None:
        return
    assert (
        "mixed_correctness_mult_reward_func" in grpo_config.reward_functions
        or "mixed_correctness_add_reward_func" in grpo_config.reward_functions
        or "mixed_additive_proposal_reward_func" in grpo_config.reward_functions
        or "mixed_multiplicative_budget_reward_func" in grpo_config.reward_functions
    )


def _load_dataset_for_config(grpo_config, split: str):
    if grpo_config.dataset in {"mbpp", "humaneval"}:
        raise ValueError(
            f"Training not supported for {grpo_config.dataset}. "
            "This dataset is evaluation-only."
        )
    if grpo_config.dataset == "gsm8k":
        return get_gsm8k_questions(split)
    if grpo_config.dataset == "math":
        return get_math_questions(split)
    if grpo_config.dataset == "gsm8k_and_math":
        _validate_mixed_reward_functions(grpo_config)
        return get_gsm8k_and_math_questions(split, seed=grpo_config.seed)
    if grpo_config.dataset == "gsm8k_and_math_and_kodcode":
        _validate_mixed_reward_functions(grpo_config)
        return get_gsm8k_and_math_and_kodcode_questions(
            split, seed=grpo_config.seed
        )
    if grpo_config.dataset == "kodcode":
        return get_kodcode_questions()
    raise ValueError(f"Dataset {grpo_config.dataset} not supported")


def _shuffle_and_limit_dataset(dataset, seed: int, max_samples: int | None, name: str):
    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        sample_count = min(int(max_samples), len(dataset))
        dataset = dataset.select(range(sample_count))
    print(f"{name} samples: {len(dataset)}")
    return dataset


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    # During training, remasking must always be "policy"
    assert grpo_config.remasking == "policy", (
        f"Training only supports remasking='policy', got '{grpo_config.remasking}'"
    )

    assert grpo_config.per_device_train_batch_size % grpo_config.num_generations == 0, (
        f"per_device_train_batch_size ({grpo_config.per_device_train_batch_size}) must be "
        f"divisible by num_generations ({grpo_config.num_generations}) to ensure complete groups per GPU."
    )

    # ES (Expert Steering) currently only supports 1 group per GPU (generates samples for first prompt only)
    if grpo_config.es_thresholds:
        assert grpo_config.num_generations == grpo_config.per_device_train_batch_size, (
            "ES requires exactly 1 group per GPU (num_generations == per_device_train_batch_size)"
        )
        assert grpo_config.block_length == 256
        assert grpo_config.policy_full_context

    validate_reward_type_matches_functions(grpo_config)
    reward_functions = get_reward_functions(grpo_config)
    train_set = _shuffle_and_limit_dataset(
        _load_dataset_for_config(grpo_config, "train"),
        grpo_config.seed,
        grpo_config.max_train_samples,
        "Train",
    )
    eval_set = None
    if grpo_config.max_eval_samples is not None:
        eval_set = _shuffle_and_limit_dataset(
            _load_dataset_for_config(grpo_config, "test"),
            grpo_config.seed,
            grpo_config.max_eval_samples,
            "Eval",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(
            "CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}; "
            f"using process cuda:{current_device} "
            f"({torch.cuda.get_device_name(current_device)})"
        )
    else:
        print("CUDA is not available; using CPU.")

    # 4 bit quantization configuration (only if enabled in ModelConfig)
    # For the paper, we left this turned off.
    bnb_config = None
    if model_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model and tokenizer
    if "LLaDA" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["LLaDA"]
        grpo_config.model_type = "LLaDA"
    elif "Dream" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["Dream"]
        grpo_config.model_type = "Dream"
    else:
        raise ValueError(f"Model path {grpo_config.model_path} not supported")

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Create policy based on type
    if grpo_config.policy_type == "dit_hidden":
        assert grpo_config.model_type == "LLaDA", (
            "dit_hidden policy is only supported with LLaDA models, not Dream"
        )
        policy_core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            num_blocks=grpo_config.policy_num_blocks,
            smart_init=grpo_config.policy_smart_init,
            time_period=grpo_config.policy_time_period,
        ).to(device)

    elif grpo_config.policy_type == "dit_confidence":
        hidden_dim = grpo_config.policy_hidden_dim or 128
        feedforward_dim = grpo_config.policy_feedforward_dim or (4 * hidden_dim)
        policy_extra_feature_names = get_policy_extra_feature_names(grpo_config)

        policy_core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=grpo_config.policy_num_heads,
            dropout=grpo_config.policy_dropout,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            smart_init=grpo_config.policy_smart_init,
            confidences_top_p=grpo_config.confidences_top_p,
            extra_feature_dim=len(policy_extra_feature_names),
            num_blocks=grpo_config.policy_num_blocks,
            time_period=grpo_config.policy_time_period,
        ).to(device)
    else:
        raise ValueError(
            f"Policy type {grpo_config.policy_type} not supported. "
            "Choose from ['dit_hidden', 'dit_confidence']"
        )

    policy = PolicyHFWrapper(policy_core, grpo_config.policy_type)
    loaded_policy_checkpoint = {
        "policy_checkpoint_path_effective": None,
        "policy_checkpoint_loaded": False,
        "policy_checkpoint_missing_keys": [],
        "policy_checkpoint_unexpected_keys": [],
    }
    if grpo_config.policy_checkpoint_path:
        loaded_policy_checkpoint = _load_policy_checkpoint(
            policy,
            grpo_config.policy_checkpoint_path,
            device,
        )
        grpo_config.policy_checkpoint_path_effective = loaded_policy_checkpoint[
            "policy_checkpoint_path_effective"
        ]
        grpo_config.policy_checkpoint_loaded = loaded_policy_checkpoint[
            "policy_checkpoint_loaded"
        ]
        grpo_config.policy_checkpoint_missing_keys = loaded_policy_checkpoint[
            "policy_checkpoint_missing_keys"
        ]
        grpo_config.policy_checkpoint_unexpected_keys = loaded_policy_checkpoint[
            "policy_checkpoint_unexpected_keys"
        ]
        print(f"policy_checkpoint_path: {grpo_config.policy_checkpoint_path}")

    # Log policy parameter count
    total_params = sum(p.numel() for p in policy_core.parameters())
    trainable_params = sum(
        p.numel() for p in policy_core.parameters() if p.requires_grad
    )

    print(f"Policy type: {grpo_config.policy_type}")
    print(f"Total policy parameters: {total_params:,}")
    print(f"Trainable policy parameters: {trainable_params:,}")

    if wandb.run is not None:
        wandb.log(
            {
                "policy/total_parameters": total_params,
                "policy/trainable_parameters": trainable_params,
                "policy/policy_type": grpo_config.policy_type,
                "policy/checkpoint_path": (
                    loaded_policy_checkpoint["policy_checkpoint_path_effective"] or ""
                ),
                "policy/checkpoint_loaded": float(
                    loaded_policy_checkpoint["policy_checkpoint_loaded"]
                ),
            },
            step=0,
        )

    output_dir = grpo_config.output_dir
    s3_output = "s3" in output_dir
    if s3_output:
        # For remote paths we save checkpoints in a temp dir locally and then
        # use a callback to push them to aws
        context_manager = tempfile.TemporaryDirectory()
        callbacks = [S3UploadCallback(output_dir)]
    else:
        # Otherwise use a context that is basically a no-op, so
        # checkpoints will be written to the path indicated by output_dir
        context_manager = nullcontext(output_dir)
        callbacks = []
        # Do however need to make sure that the local path exists!
        os.makedirs(output_dir, exist_ok=True)

    with context_manager as local_output_dir:
        grpo_config.output_dir = local_output_dir

        # Check for existing checkpoint to resume from
        resume_from = None
        if s3_output:
            latest = get_latest_s3_checkpoint(output_dir)
            if latest:
                resume_from = download_s3_checkpoint(
                    output_dir, latest, local_output_dir
                )
                print(
                    f"=== Auto-resume: found {latest}, downloaded to {resume_from} ==="
                )

        trainer = Trainer(
            args=grpo_config,
            model=policy,
            dllm=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=eval_set,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train(resume_from_checkpoint=resume_from)

        if resume_from:
            print(f"=== Resumed training from step {trainer.state.global_step} ===")
        else:
            print(
                f"=== Started fresh training, now at step {trainer.state.global_step} ==="
            )


if __name__ == "__main__":
    parser = TrlParser((Config, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )
    main(grpo_config=grpo_config, model_config=model_config)
