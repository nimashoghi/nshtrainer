import contextlib
import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nshconfig as C
from nshrunner._env import SNAPSHOT_DIR
from typing_extensions import override

from ._callback import NTCallbackBase
from .callbacks.base import CallbackConfigBase

if TYPE_CHECKING:
    from huggingface_hub import HfApi  # noqa: F401

    from .model.base import BaseConfig


log = logging.getLogger(__name__)


class HuggingFaceHubAutoCreateConfig(C.Config):
    enabled: bool = True
    """Enable automatic repository creation on the Hugging Face Hub."""

    private: bool = True
    """Whether to create the repository as private."""

    namespace: str | None = None
    """The namespace to create the repository in. If `None`, the repository will be created in the user's namespace."""

    def __bool__(self):
        return self.enabled


class HuggingFaceHubConfig(CallbackConfigBase):
    """Configuration options for Hugging Face Hub integration."""

    enabled: bool = False
    """Enable Hugging Face Hub integration."""

    token: str | None = None
    """Hugging Face Hub API token. If `None`, the token will be read from the current environment.
    This needs to either be set using `huggingface-cli login` or by setting the `HUGGINGFACE_TOKEN`
    environment variable."""

    auto_create: HuggingFaceHubAutoCreateConfig = HuggingFaceHubAutoCreateConfig()
    """Automatic repository creation configuration options."""

    save_config: bool = True
    """Whether to save the model configuration to the Hugging Face Hub."""

    save_checkpoints: bool = True
    """Whether to save checkpoints to the Hugging Face Hub."""

    save_code: bool = True
    """Whether to save code to the Hugging Face Hub.
    This is only supported if `nshsnap` is installed and snapshotting is enabled."""

    save_in_background: bool = True
    """Whether to save to the Hugging Face Hub in the background.
    This corresponds to setting `run_as_future=True` in the HFApi upload methods."""

    def enable_(self):
        self.enabled = True
        return self

    def disable_(self):
        self.enabled = False
        return self

    def __bool__(self):
        return self.enabled

    @override
    def create_callbacks(self, root_config):
        yield self.with_metadata(HFHubCallback(self), ignore_if_exists=True)


def _api(token: str | None = None):
    # Make sure that `huggingface_hub` is installed
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log.exception(
            "Could not import `huggingface_hub`. Please install it using `pip install huggingface_hub`."
        )
        return None

    # Create and authenticate the API instance
    try:
        api = huggingface_hub.HfApi(token=token)

        # Verify authentication
        api.whoami()
    except Exception:
        log.exception(
            "Authentication failed for Hugging Face Hub. "
            "Please make sure you are logged in using `huggingface-cli login`, "
            "by setting the HUGGING_FACE_HUB_TOKEN environment variable, "
            "or by providing a valid token in the configuration."
        )
        return None

    return api


def _repo_name(api: "HfApi", root_config: "BaseConfig"):
    username = None
    if (ac := root_config.trainer.hf_hub.auto_create) and ac.namespace:
        username = ac.namespace
    elif (username := api.whoami().get("name", None)) is None:
        raise ValueError("Could not get username from Hugging Face Hub.")

    # Sanitize the project (if it exists), run_name, and id
    parts = []
    if root_config.project:
        parts.append(re.sub(r"[^a-zA-Z0-9-]", "-", root_config.project))
    parts.append(re.sub(r"[^a-zA-Z0-9-]", "-", root_config.run_name))
    parts.append(re.sub(r"[^a-zA-Z0-9-]", "-", root_config.id))

    # Combine parts and ensure it starts and ends with alphanumeric characters
    repo_name = "-".join(parts)
    repo_name = repo_name.strip("-")
    repo_name = re.sub(
        r"-+", "-", repo_name
    )  # Replace multiple dashes with a single dash

    # Ensure the name is not longer than 96 characters (excluding username)
    if len(repo_name) > 96:
        repo_name = repo_name[:96].rstrip("-")

    # Ensure the repo name starts with an alphanumeric character
    repo_name = re.sub(r"^[^a-zA-Z0-9]+", "", repo_name)

    # If the repo_name is empty after all sanitization, use a default name
    if not repo_name:
        repo_name = "default-repo-name"

    return f"{username}/{repo_name}"


@dataclass
class _Upload:
    local_path: Path
    path_in_repo: Path

    @classmethod
    def from_local_path(
        cls,
        local_path: Path,
        root_config: "BaseConfig",
    ):
        # Resolve the checkpoint directory
        checkpoint_dir = root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        try:
            relative_path = local_path.relative_to(checkpoint_dir)
        except ValueError:
            raise ValueError(
                f"Checkpoint path {local_path} is not within the checkpoint directory {checkpoint_dir}."
            )

        # Prefix the path in repo with "checkpoints"
        path_in_repo = Path("checkpoints") / relative_path

        return cls(local_path=local_path, path_in_repo=path_in_repo)


class HFHubCallback(NTCallbackBase):
    @contextlib.contextmanager
    def _with_error_handling(self, opeartion: str):
        try:
            yield
        except Exception:
            log.exception(f"Failed to {opeartion}, repo_id={self._repo_id}")
        else:
            log.debug(f"Successfully {opeartion}, repo_id={self._repo_id}")

    def __init__(self, config: HuggingFaceHubConfig):
        super().__init__()

        self.config = config

        self._repo_id: str | None = None
        self._checksum_to_path_in_repo: dict[str, Path] = {}

    @override
    def setup(self, trainer, pl_module, stage):
        root_config = cast("BaseConfig", pl_module.hparams)
        self._repo_id = _repo_name(self.api, root_config)

        if not self.config or not trainer.is_global_zero:
            return

        # Create the repository, if it doesn't exist
        self._create_repo_if_not_exists()

        # Upload the config and code
        self._save_config(root_config)
        self._save_code()

    @override
    def on_checkpoint_saved(self, ckpt_path, metadata_path, trainer, pl_module):
        # If HF Hub is enabled, then we upload
        if (
            not self.config
            or not self.config.save_checkpoints
            or not trainer.is_global_zero
        ):
            return

        with self._with_error_handling("save checkpoints"):
            root_config = cast("BaseConfig", pl_module.hparams)
            self._save_checkpoint(
                _Upload.from_local_path(ckpt_path, root_config),
                _Upload.from_local_path(metadata_path, root_config)
                if metadata_path is not None
                else None,
            )

    @cached_property
    def api(self):
        # Create and authenticate the API instance
        if (api := _api(self.config.token)) is None:
            raise ValueError("Failed to create Hugging Face Hub API instance.")
        return api

    @property
    def repo_id(self):
        if self._repo_id is None:
            raise ValueError("Repository id has not been initialized.")
        return self._repo_id

    def _create_repo_if_not_exists(self):
        if not self.config or not self.config.auto_create:
            return

        # Create the repository, if it doesn't exist
        with self._with_error_handling("create repository"):
            from huggingface_hub.utils import RepositoryNotFoundError

            try:
                # Check if the repository exists
                self.api.repo_info(repo_id=self.repo_id, repo_type="model")
                log.info(f"Repository '{self.repo_id}' already exists.")
            except RepositoryNotFoundError:
                # Repository doesn't exist, so create it
                try:
                    self.api.create_repo(
                        repo_id=self.repo_id,
                        repo_type="model",
                        private=self.config.auto_create.private,
                        exist_ok=True,
                    )
                    log.info(f"Created new repository '{self.repo_id}'.")
                except Exception:
                    log.exception(f"Failed to create repository '{self.repo_id}'")
            except Exception:
                log.exception(f"Error checking repository '{self.repo_id}'")

    def _save_config(self, root_config: "BaseConfig"):
        with self._with_error_handling("upload config"):
            self.api.upload_file(
                path_or_fileobj=root_config.model_dump_json(indent=4).encode("utf-8"),
                path_in_repo="config.json",
                repo_id=self.repo_id,
                repo_type="model",
                run_as_future=cast(Any, self.config.save_in_background),
            )

    def _save_code(self):
        # If a snapshot has been taken (which can be detected using the SNAPSHOT_DIR env),
        # then upload all contents within the snapshot directory to the repository.
        if not (snapshot_dir := os.environ.get(SNAPSHOT_DIR)):
            log.debug("No snapshot directory found. Skipping upload.")
            return

        with self._with_error_handling("save code"):
            snapshot_dir = Path(snapshot_dir)
            if not snapshot_dir.exists() or not snapshot_dir.is_dir():
                log.warning(
                    f"Snapshot directory '{snapshot_dir}' does not exist or is not a directory."
                )
                return

            self.api.upload_folder(
                folder_path=str(snapshot_dir),
                repo_id=self.repo_id,
                repo_type="model",
                path_in_repo="code",  # Prefix with "code" folder
                run_as_future=cast(Any, self.config.save_in_background),
            )

    def _save_file(self, p: _Upload):
        with self._with_error_handling("save file"):
            # Upload the checkpoint files to the repository
            self.api.upload_file(
                path_or_fileobj=p.local_path,
                path_in_repo=str(p.path_in_repo),
                repo_id=self.repo_id,
                repo_type="model",
                run_as_future=cast(Any, self.config.save_in_background),
            )

    def _copy_file(self, source_path_in_repo: Path, dest_path_in_repo: Path):
        # Create a commit for copying the files
        from huggingface_hub.hf_api import CommitOperationCopy

        with self._with_error_handling("copy file"):
            copy_op = CommitOperationCopy(
                src_path_in_repo=str(source_path_in_repo),
                path_in_repo=str(dest_path_in_repo),
            )

            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                commit_message="Copy checkpoint file",
                operations=[copy_op],
                run_as_future=cast(Any, self.config.save_in_background),
            )

    def _save_checkpoint(self, path: _Upload, metadata_path: _Upload | None):
        if not self.config.save_checkpoints:
            return

        # If no metadata, just save regularly.
        if metadata_path is None:
            self._save_file(path)
            return

        # Otherwise, let's check to see if we've already uploaded the metadata.
        # If so, we can just copy the checkpoint file.
        from ._checkpoint.metadata import CheckpointMetadata

        metadata = CheckpointMetadata.from_file(metadata_path.local_path)
        if (
            existing_ckpt_path := self._checksum_to_path_in_repo.get(
                metadata.checkpoint_checksum
            )
        ) is not None:
            self._copy_file(existing_ckpt_path, path.path_in_repo)
        else:
            # Otherwise, we save the checkpoint & keep the checksum so we don't
            # re-upload the same file again.
            self._save_file(path)
            self._checksum_to_path_in_repo[metadata.checkpoint_checksum] = (
                path.path_in_repo
            )

        # Save the metadata file
        # NOTE: This file is fairly small, so we can just upload it directly.
        # No need to copy.
        self._save_file(metadata_path)
