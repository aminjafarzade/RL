#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import logging
import os
from pathlib import Path

import s3fs
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


def configure_s3(output_path: Path) -> s3fs.S3FileSystem:
    raise NotImplementedError("Internal bucket setup stripped.")


def get_latest_s3_checkpoint(remote_path: str) -> str | None:
    s3 = configure_s3(remote_path)
    bucket_path = remote_path.replace("s3://", "")

    try:
        entries = s3.ls(bucket_path)
    except FileNotFoundError:
        return None

    latest_step, latest_name = -1, None
    for entry in entries:
        name = entry.split("/")[-1]
        if not name.startswith("checkpoint-") or name == "checkpoint-best":
            continue
        try:
            step = int(name.split("-")[1])
        except ValueError:
            continue
        if step > latest_step:
            latest_step, latest_name = step, name

    return latest_name


def download_s3_checkpoint(
    remote_path: str, checkpoint_name: str, local_dir: str
) -> str:
    s3 = configure_s3(remote_path)
    bucket_path = remote_path.replace("s3://", "")
    remote = f"{bucket_path}/{checkpoint_name}"
    local = os.path.join(local_dir, checkpoint_name)
    logger.info(f"Downloading checkpoint: {remote} -> {local}")
    s3.get(remote, local, recursive=True)
    return local


class S3UploadCallback(TrainerCallback):
    def __init__(self, remote_path: Path):
        self.s3 = configure_s3(remote_path)
        self.remote_path = remote_path

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        best: bool = False,
        **kwargs,
    ):
        """Event called after a checkpoint is saved locally."""

        if best:
            checkpoint_name = Path("checkpoint-best")
        else:
            # The latest checkpoint is saved in a subfolder of output_dir
            # e.g., in /tmp/tmpxxxxxx/checkpoint-500
            checkpoint_name = Path(f"checkpoint-{state.global_step}")
        local_checkpoint_dir = args.output_dir / checkpoint_name

        if local_checkpoint_dir:
            remote_checkpoint_dir = f"{self.remote_path}/{checkpoint_name}"

            logger.info(
                f"Callback triggered: Uploading '{local_checkpoint_dir}' to '{remote_checkpoint_dir}'..."
            )

            try:
                self.s3.put(local_checkpoint_dir, remote_checkpoint_dir, recursive=True)
                logging.info(
                    f"Successfully uploaded checkpoint: {local_checkpoint_dir} -> {remote_checkpoint_dir}"
                )
            except Exception as e:
                logging.error(
                    f"s3 failed to upload {local_checkpoint_dir} -> {remote_checkpoint_dir}: {e}"
                )
