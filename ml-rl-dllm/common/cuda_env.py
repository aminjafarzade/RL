#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Early CUDA visibility selection.

This module intentionally avoids importing torch/accelerate/transformers. Call
``set_cuda_visible_devices_from_argv`` before any CUDA-aware library import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from typing import Sequence


_NONE_VALUES = {"", "none", "null"}


def _strip_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _normalize_cuda_visible_devices(value: str) -> str:
    value = _strip_scalar(value)
    if value.lower() in _NONE_VALUES:
        return ""
    if value.startswith("[") and value.endswith("]"):
        value = ",".join(
            _strip_scalar(part) for part in value[1:-1].split(",") if part.strip()
        )
    return value.replace(" ", "")


def _normalize_gpu_alias(value: str) -> str:
    value = _normalize_cuda_visible_devices(value)
    if value.lower().startswith("cuda:"):
        value = value.split(":", 1)[1]
    if value.lower().startswith("gpu:"):
        value = value.split(":", 1)[1]
    return value


def _option_value(argv: Sequence[str], names: Iterable[str]) -> str | None:
    names = tuple(names)
    for idx, arg in enumerate(argv):
        for name in names:
            if arg == name and idx + 1 < len(argv):
                return argv[idx + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


def _find_cli_config_path(argv: Sequence[str], names: Iterable[str]) -> str | None:
    return _option_value(argv, names)


def _read_top_level_config_values(
    config_path: str | Path,
    keys: set[str],
) -> dict[str, str]:
    values = {}
    with Path(config_path).open(encoding="utf-8") as config_file:
        for raw_line in config_file:
            if raw_line[:1].isspace():
                continue
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if key in keys:
                values[key] = raw_value.split("#", 1)[0].strip()
    return values


def _parse_optional_float(value: str, key: str) -> float | None:
    value = _strip_scalar(value)
    if value.lower() in _NONE_VALUES:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def _query_cuda_devices_by_memory() -> list[tuple[str, float, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cuda_min_memory_gb requires nvidia-smi to select a GPU by memory."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while querying GPUs with nvidia-smi.") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed while selecting a GPU: {stderr}")

    devices = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            devices.append((parts[0], float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not devices:
        raise RuntimeError("nvidia-smi returned no parseable GPU memory rows.")
    return devices


def _format_cuda_devices(devices: list[tuple[str, float, float]]) -> str:
    return ", ".join(
        f"{index}: total={total_mib / 1024:.1f}GiB free={free_mib / 1024:.1f}GiB"
        for index, total_mib, free_mib in devices
    )


def _device_index_for_sort(index: str) -> int:
    return int(index) if index.isdigit() else 0


def _select_cuda_device_with_min_memory(min_memory_gb: float) -> str:
    devices = _query_cuda_devices_by_memory()
    min_memory_mib = min_memory_gb * 1024
    eligible = [
        (index, total_mib, free_mib)
        for index, total_mib, free_mib in devices
        if total_mib >= min_memory_mib
    ]
    if not eligible:
        raise RuntimeError(
            f"No CUDA GPU has at least {min_memory_gb:g}GiB total memory. "
            f"Available GPUs: {_format_cuda_devices(devices)}"
        )

    eligible.sort(
        key=lambda item: (item[1], -item[2], _device_index_for_sort(item[0]))
    )
    selected_index, total_mib, free_mib = eligible[0]
    print(
        "Selected CUDA GPU "
        f"{selected_index} for cuda_min_memory_gb={min_memory_gb:g} "
        f"(total={total_mib / 1024:.1f}GiB, free={free_mib / 1024:.1f}GiB)."
    )
    return selected_index


def _set_cuda_visible_devices(value: str, source: str, verbose: bool) -> bool:
    if not value:
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    if verbose:
        print(
            f"CUDA_VISIBLE_DEVICES={value} selected from {source}; "
            "the process will see this as cuda:0."
        )
    return True


def set_cuda_visible_devices_from_argv(
    argv: Sequence[str] | None = None,
    *,
    config_arg_names: Sequence[str] = ("--config",),
    config_path: str | None = None,
    respect_existing: bool = True,
    verbose: bool = True,
) -> None:
    """Set CUDA_VISIBLE_DEVICES from CLI/config before torch is imported.

    Precedence:
    1. CLI --cuda_visible_devices.
    2. CLI --gpu.
    3. Existing CUDA_VISIBLE_DEVICES environment, when respect_existing=True.
    4. Top-level YAML cuda_visible_devices.
    5. Top-level YAML gpu.
    6. CLI/YAML cuda_min_memory_gb auto-selection.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    cli_visible = _option_value(
        argv,
        ("--cuda_visible_devices", "--cuda-visible-devices"),
    )
    if cli_visible is not None:
        if _set_cuda_visible_devices(
            _normalize_cuda_visible_devices(cli_visible),
            "--cuda_visible_devices",
            verbose,
        ):
            return

    cli_gpu = _option_value(argv, ("--gpu",))
    if cli_gpu is not None:
        if _set_cuda_visible_devices(_normalize_gpu_alias(cli_gpu), "--gpu", verbose):
            return

    if respect_existing and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    expanded_config_path = config_path or _find_cli_config_path(
        argv,
        config_arg_names,
    )
    values: dict[str, str] = {}
    if expanded_config_path:
        expanded = os.path.expandvars(os.path.expanduser(expanded_config_path))
        if os.path.exists(expanded):
            values = _read_top_level_config_values(
                expanded,
                {"cuda_visible_devices", "gpu", "cuda_min_memory_gb"},
            )

    config_visible = values.get("cuda_visible_devices")
    if config_visible is not None:
        if _set_cuda_visible_devices(
            _normalize_cuda_visible_devices(config_visible),
            "config cuda_visible_devices",
            verbose,
        ):
            return

    config_gpu = values.get("gpu")
    if config_gpu is not None:
        if _set_cuda_visible_devices(
            _normalize_gpu_alias(config_gpu),
            "config gpu",
            verbose,
        ):
            return

    cli_min_memory = _option_value(
        argv,
        ("--cuda_min_memory_gb", "--cuda-min-memory-gb"),
    )
    min_memory = cli_min_memory or values.get("cuda_min_memory_gb")
    if min_memory is not None:
        parsed_min_memory = _parse_optional_float(min_memory, "cuda_min_memory_gb")
        if parsed_min_memory is not None:
            _set_cuda_visible_devices(
                _select_cuda_device_with_min_memory(parsed_min_memory),
                "cuda_min_memory_gb",
                verbose,
            )
