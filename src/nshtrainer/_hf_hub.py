import io
import logging
import os
import re
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
    from .trainer.trainer import Trainer


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


def _enabled_and_valid(
    trainer: "Trainer",
    config: HuggingFaceHubConfig,
    *,
    rank_zero_only: bool,
):
    # Make sure this is enabled and the config is valid
    if not config:
        return None

    # If `rank_zero_only` and this is not rank 0, stop here.
    if rank_zero_only and not trainer.is_global_zero:
        return

    # Make sure that `huggingface_hub` is installed
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log.exception(
            "Could not import `huggingface_hub`. Please install it using `pip install huggingface_hub`."
        )
        return None

    # Create and authenticate the API instance
    if (api := getattr(trainer, "_hf_hub_api", None)) is None:
        api = _api(config.token)
        setattr(trainer, "_hf_hub_api", api)
    return cast(huggingface_hub.HfApi, api)


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


def _init(*, trainer: "Trainer", root_config: "BaseConfig"):
    config = root_config.trainer.hf_hub
    if (
        api := _enabled_and_valid(
            trainer,
            config,
            rank_zero_only=True,
        )
    ) is None or not config.auto_create:
        return

    from huggingface_hub.utils import RepositoryNotFoundError

    # Resolve the repository name
    repo_name = _repo_name(api, root_config)

    # Create the repository, if it doesn't exist
    try:
        # Check if the repository exists
        api.repo_info(repo_id=repo_name, repo_type="model")
        log.info(f"Repository '{repo_name}' already exists.")
    except RepositoryNotFoundError:
        # Repository doesn't exist, so create it
        try:
            api.create_repo(
                repo_id=repo_name,
                repo_type="model",
                private=config.auto_create.private,
                exist_ok=True,
            )
            log.info(f"Created new repository '{repo_name}'.")
        except Exception:
            log.exception(f"Failed to create repository '{repo_name}'")
    except Exception:
        log.exception(f"Error checking repository '{repo_name}'")

    # Upload the config
    _save_config(root_config, trainer=trainer)

    # Upload the code
    _save_code(repo_name, config=config, trainer=trainer)


def _save_code(
    repo_name: str,
    *,
    config: HuggingFaceHubConfig,
    trainer: "Trainer",
):
    if (
        api := _enabled_and_valid(
            trainer,
            config,
            rank_zero_only=True,
        )
    ) is None or not config.save_code:
        return

    # If a snapshot has been taken (which can be detected using the SNAPSHOT_DIR env),
    # then upload all contents within the snapshot directory to the repository.
    snapshot_dir = os.environ.get(SNAPSHOT_DIR)
    if not snapshot_dir:
        log.info("No snapshot directory found. Skipping upload.")
        return

    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.exists() or not snapshot_path.is_dir():
        log.warning(
            f"Snapshot directory '{snapshot_dir}' does not exist or is not a directory."
        )
        return

    try:
        api.upload_folder(
            folder_path=str(snapshot_path),
            repo_id=repo_name,
            repo_type="model",
            path_in_repo="code",  # Prefix with "code" folder
            run_as_future=cast(Any, config.save_in_background),
        )
        log.info(
            f"Uploaded snapshot contents to repository '{repo_name}' under 'code' folder."
        )
    except Exception:
        log.exception(
            f"Failed to upload snapshot contents to repository '{repo_name}' under 'code' folder."
        )


def _save_config(
    root_config: "BaseConfig",
    *,
    trainer: "Trainer",
):
    config = root_config.trainer.hf_hub
    if (
        api := _enabled_and_valid(
            trainer,
            config,
            rank_zero_only=True,
        )
    ) is None or not config.save_config:
        return

    # Convert the root config to a JSON string
    # NOTE: This is a utf-8 string.
    config_json = root_config.model_dump_json(indent=4)

    # Resolve the repository name
    repo_name = _repo_name(api, root_config)

    # Upload the config file to the repository
    try:
        api.upload_file(
            path_or_fileobj=config_json.encode("utf-8"),
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
            run_as_future=cast(Any, config.save_in_background),
        )
        log.info(f"Uploaded config.json to repository '{repo_name}'.")
    except Exception:
        log.exception(f"Failed to upload config.json to repository '{repo_name}'.")


def _save_checkpoint_files(
    trainer: "Trainer",
    paths: list[Path],
    *,
    root_config: "BaseConfig",
):
    config = root_config.trainer.hf_hub
    if (
        api := _enabled_and_valid(trainer, config, rank_zero_only=True)
    ) is None or not config.save_checkpoints:
        return

    # Resolve the checkpoint directory
    checkpoint_dir = root_config.directory.resolve_subdirectory(
        root_config.id, "checkpoint"
    )

    # Resolve the repository name
    repo_name = _repo_name(api, root_config)

    # Let's read all the files to memory right now,
    # in case they get used/removed by other processes.
    # Read all the files to memory
    file_contents: list[bytes | None] = []
    for p in paths:
        assert not p.is_symlink(), f"Path {p} is a symlink."
        assert p.is_file(), f"Path {p} is not a file."
        try:
            with open(p, "rb") as f:
                file_contents.append(f.read())
        except IOError:
            log.exception(f"Failed to read checkpoint file {p}.")
            file_contents.append(None)

    # Remove the paths that failed to read
    file_contents_and_paths = [
        (contents, p)
        for contents, p in zip(file_contents, paths)
        if contents is not None
    ]

    # Upload the checkpoint files to the repository
    for contents, p in file_contents_and_paths:
        try:
            relative_path = p.relative_to(checkpoint_dir)
        except ValueError:
            log.warning(
                f"Checkpoint path {p} is not within the checkpoint directory {checkpoint_dir}."
            )
            continue

        # Prefix the path in repo with "checkpoints"
        path_in_repo = Path("checkpoints") / relative_path

        # Upload the checkpoint file to the repository
        try:
            api.upload_file(
                path_or_fileobj=io.BytesIO(contents),
                path_in_repo=str(path_in_repo),
                repo_id=repo_name,
                repo_type="model",
                run_as_future=cast(Any, config.save_in_background),
            )
            log.info(
                f"Uploaded checkpoint file {relative_path} to repository '{repo_name}'."
            )
        except Exception:
            log.exception(
                f"Failed to upload checkpoint file {relative_path} to repository '{repo_name}'."
            )

    log.info(f"Completed uploading checkpoint files to repository '{repo_name}'.")


def _save_checkpoint_symlinks(
    trainer: "Trainer",
    paths: list[Path],
    *,
    root_config: "BaseConfig",
):
    config = root_config.trainer.hf_hub
    if (
        api := _enabled_and_valid(trainer, config, rank_zero_only=True)
    ) is None or not config.save_checkpoints:
        return

    # Resolve the checkpoint directory
    checkpoint_dir = root_config.directory.resolve_subdirectory(
        root_config.id, "checkpoint"
    )

    # Resolve the repository name
    repo_name = _repo_name(api, root_config)

    # Create a commit for copying the files
    from huggingface_hub.hf_api import CommitOperation, CommitOperationCopy

    commits: list[CommitOperation] = []
    for p in paths:
        assert p.is_symlink(), f"Path {p} is not a symlink."

        try:
            dest_relative_path = p.relative_to(checkpoint_dir)
        except ValueError:
            log.warning(
                f"Checkpoint path {p} is not within the checkpoint directory {checkpoint_dir}."
            )
            continue

        try:
            source_relative_path = p.resolve().relative_to(checkpoint_dir)
        except ValueError:
            log.warning(
                f"Checkpoint symlink target {p.resolve()} is not within the checkpoint directory {checkpoint_dir}."
            )
            continue

        # Prefix the path in repo with "checkpoints"
        dest_path_in_repo = Path("checkpoints") / dest_relative_path
        source_path_in_repo = Path("checkpoints") / source_relative_path

        # Create and append a CommitOperationCopy for copying the symlink
        copy_op = CommitOperationCopy(
            src_path_in_repo=str(source_path_in_repo),
            path_in_repo=str(dest_path_in_repo),
        )
        commits.append(copy_op)

    log.info(f"Creating a commit with the following operations: {commits}")

    try:
        api.create_commit(
            repo_id=repo_name,
            repo_type="model",
            commit_message="Copy checkpoint symlinks",
            operations=commits,
            run_as_future=cast(Any, config.save_in_background),
        )
        log.info(
            f"Created commit to copy checkpoint symlinks to repository '{repo_name}'."
        )
    except Exception:
        log.exception(
            f"Failed to create commit to copy checkpoint symlinks to repository '{repo_name}'"
        )

    log.info(f"Completed copying checkpoint symlinks to repository '{repo_name}'.")


def _save_checkpoint_directory(trainer: "Trainer", *, root_config: "BaseConfig"):
    config = root_config.trainer.hf_hub
    if (
        api := _enabled_and_valid(trainer, config, rank_zero_only=True)
    ) is None or not config.save_checkpoints:
        return

    # Resolve the checkpoint directory
    checkpoint_dir = root_config.directory.resolve_subdirectory(
        root_config.id, "checkpoint"
    )

    # Resolve the repository name
    repo_name = _repo_name(api, root_config)

    # Upload the checkpoint directory to the repository
    try:
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_name,
            repo_type="model",
            path_in_repo="checkpoints",
            run_as_future=cast(Any, config.save_in_background),
        )
        log.info(f"Uploaded checkpoint directory to repository '{repo_name}'.")
    except Exception:
        log.exception(
            f"Failed to upload checkpoint directory to repository '{repo_name}'."
        )

    log.info(f"Completed uploading checkpoint files to repository '{repo_name}'.")


class HFHubCallback(NTCallbackBase):
    def __init__(self, config: HuggingFaceHubConfig):
        super().__init__()
        self.config = config

    @override
    def setup(self, trainer, pl_module, stage):
        from .trainer.trainer import Trainer

        if not isinstance(trainer, Trainer):
            raise ValueError(
                f"HFHubCallback requires a `nshtrainer.Trainer` instance, got {type(trainer)}."
            )

        root_config = cast("BaseConfig", pl_module.hparams)
        _init(trainer=trainer, root_config=root_config)

    @override
    def teardown(self, trainer, pl_module, stage):
        if hasattr(trainer, "_hf_hub_api"):
            delattr(trainer, "_hf_hub_api")

    @override
    def on_checkpoint_saved(self, ckpt_path, metadata_path, trainer, pl_module):
        root_config = cast("BaseConfig", pl_module.hparams)

        # If HF Hub is enabled, then we upload
        if root_config.trainer.hf_hub and trainer.is_global_zero:
            # Upload the regular files first, then the symlinks
            all_paths = [p for p in (ckpt_path, metadata_path) if p is not None]
            if regular_paths := [p for p in all_paths if not p.is_symlink()]:
                _save_checkpoint_files(trainer, regular_paths, root_config=root_config)
            if symlink_paths := [p for p in all_paths if p.is_symlink()]:
                _save_checkpoint_symlinks(
                    trainer, symlink_paths, root_config=root_config
                )
