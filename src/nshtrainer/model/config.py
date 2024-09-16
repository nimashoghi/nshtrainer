import copy
import logging
import os
import string
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, ClassVar

import nshconfig as C
import numpy as np
import torch
from typing_extensions import Self

from ..callbacks.base import CallbackConfigBase
from ..loggers import LoggerConfig
from ..metrics import MetricConfig
from ..trainer._config import TrainerConfig
from ..util._environment_info import EnvironmentConfig

log = logging.getLogger(__name__)


def _create_symlink_to_nshrunner(base_dir: Path):
    # Resolve the current nshrunner session directory
    if not (session_dir := os.environ.get("NSHRUNNER_SESSION_DIR")):
        log.warning("NSHRUNNER_SESSION_DIR is not set. Skipping symlink creation.")
        return
    session_dir = Path(session_dir)
    if not session_dir.exists() or not session_dir.is_dir():
        log.warning(
            f"NSHRUNNER_SESSION_DIR is not a valid directory: {session_dir}. "
            "Skipping symlink creation."
        )
        return

    # Create the symlink
    symlink_path = base_dir / "nshrunner"
    if symlink_path.exists():
        # If it already points to the correct directory, we're done
        if symlink_path.resolve() == session_dir.resolve():
            return

        # Otherwise, we should log a warning and remove the existing symlink
        log.warning(
            f"A symlink pointing to {symlink_path.resolve()} already exists at {symlink_path}. "
            "Removing the existing symlink."
        )
        symlink_path.unlink()

    symlink_path.symlink_to(session_dir)


class DirectoryConfig(C.Config):
    project_root: Path | None = None
    """
    Root directory for this project.

    This isn't specific to the run; it is the parent directory of all runs.
    """

    create_symlink_to_nshrunner_root: bool = True
    """Should we create a symlink to the root folder for the Runner (if we're in one)?"""

    log: Path | None = None
    """Base directory for all experiment tracking (e.g., WandB, Tensorboard, etc.) files. If None, will use nshtrainer/{id}/log/."""

    stdio: Path | None = None
    """stdout/stderr log directory to use for the trainer. If None, will use nshtrainer/{id}/stdio/."""

    checkpoint: Path | None = None
    """Checkpoint directory to use for the trainer. If None, will use nshtrainer/{id}/checkpoint/."""

    activation: Path | None = None
    """Activation directory to use for the trainer. If None, will use nshtrainer/{id}/activation/."""

    profile: Path | None = None
    """Directory to save profiling information to. If None, will use nshtrainer/{id}/profile/."""

    def resolve_run_root_directory(self, run_id: str) -> Path:
        if (project_root_dir := self.project_root) is None:
            project_root_dir = Path.cwd()

        # The default base dir is $CWD/nshtrainer/{id}/
        base_dir = project_root_dir / "nshtrainer"
        base_dir.mkdir(exist_ok=True)

        # Add a .gitignore file to the nshtrainer directory
        #   which will ignore all files except for the .gitignore file itself
        gitignore_path = base_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.touch()
            gitignore_path.write_text("*\n")

        base_dir = base_dir / run_id
        base_dir.mkdir(exist_ok=True)

        # Create a symlink to the root folder for the Runner
        if self.create_symlink_to_nshrunner_root:
            _create_symlink_to_nshrunner(base_dir)

        return base_dir

    def resolve_subdirectory(
        self,
        run_id: str,
        # subdirectory: Literal["log", "stdio", "checkpoint", "activation", "profile"],
        subdirectory: str,
    ) -> Path:
        # The subdir will be $CWD/nshtrainer/{id}/{log, stdio, checkpoint, activation}/
        if (subdir := getattr(self, subdirectory, None)) is not None:
            assert isinstance(
                subdir, Path
            ), f"Expected a Path for {subdirectory}, got {type(subdir)}"
            return subdir

        dir = self.resolve_run_root_directory(run_id)
        dir = dir / subdirectory
        dir.mkdir(exist_ok=True)
        return dir

    def _resolve_log_directory_for_logger(self, run_id: str, logger: LoggerConfig):
        if (log_dir := logger.log_dir) is not None:
            return log_dir

        # Save to nshtrainer/{id}/log/{logger name}
        log_dir = self.resolve_subdirectory(run_id, "log")
        log_dir = log_dir / logger.name
        log_dir.mkdir(exist_ok=True)

        return log_dir


class BaseConfig(C.Config):
    id: str = C.Field(default_factory=lambda: BaseConfig.generate_id())
    """ID of the run."""
    name: str | None = None
    """Run name."""
    name_parts: list[str] = []
    """A list of parts used to construct the run name. This is useful for constructing the run name dynamically."""
    project: str | None = None
    """Project name."""
    tags: list[str] = []
    """Tags for the run."""
    notes: list[str] = []
    """Human readable notes for the run."""

    debug: bool = False
    """Whether to run in debug mode. This will enable debug logging and enable debug code paths."""
    environment: Annotated[EnvironmentConfig, C.Field(repr=False)] = (
        EnvironmentConfig.empty()
    )
    """A snapshot of the current environment information (e.g. python version, slurm info, etc.). This is automatically populated by the run script."""

    directory: DirectoryConfig = DirectoryConfig()
    """Directory configuration options."""
    trainer: TrainerConfig = TrainerConfig()
    """PyTorch Lightning trainer configuration options. Check Lightning's `Trainer` documentation for more information."""

    primary_metric: MetricConfig | None = None
    """Primary metric configuration options. This is used in the following ways:
    - To determine the best model checkpoint to save with the ModelCheckpoint callback.
    - To monitor the primary metric during training and stop training based on the `early_stopping` configuration.
    - For the ReduceLROnPlateau scheduler.
    """

    meta: dict[str, Any] = {}
    """Additional metadata for this run. This can be used to store arbitrary data that is not part of the config schema."""

    @property
    def run_name(self) -> str:
        parts = self.name_parts.copy()
        if self.name is not None:
            parts = [self.name] + parts
        name = "-".join(parts)
        if not name:
            name = self.id
        return name

    def clone(self, with_new_id: bool = True) -> Self:
        c = copy.deepcopy(self)
        if with_new_id:
            c.id = BaseConfig.generate_id()
        return c

    def subdirectory(self, subdirectory: str) -> Path:
        return self.directory.resolve_subdirectory(self.id, subdirectory)

    # region Helper methods
    def with_project_root_(self, project_root: str | Path | os.PathLike) -> Self:
        """
        Set the project root directory for the trainer.

        Args:
            project_root (Path): The base directory to use.

        Returns:
            self: The current instance of the class.
        """
        self.directory.project_root = Path(project_root)
        return self

    def reset_(
        self,
        *,
        id: bool = True,
        basic: bool = True,
        project_root: bool = True,
        environment: bool = True,
        meta: bool = True,
    ):
        """
        Reset the configuration object to its initial state.

        Parameters:
        - id (bool): If True, generate a new ID for the configuration object.
        - basic (bool): If True, reset basic attributes like name, project, tags, and notes.
        - project_root (bool): If True, reset the directory configuration to its initial state.
        - environment (bool): If True, reset the environment configuration to its initial state.
        - meta (bool): If True, reset the meta dictionary to an empty dictionary.

        Returns:
        - self: The updated configuration object.

        """
        if id:
            self.id = self.generate_id()

        if basic:
            self.name = None
            self.name_parts = []
            self.project = None
            self.tags = []
            self.notes = []

        if project_root:
            self.directory = DirectoryConfig()

        if environment:
            self.environment = EnvironmentConfig.empty()

        if meta:
            self.meta = {}

        return self

    def concise_repr(self) -> str:
        """Get a concise representation of the configuration object."""

        def _truncate(s: str, max_len: int = 50):
            return s if len(s) <= max_len else f"{s[:max_len - 3]}..."

        cls_name = self.__class__.__name__

        parts: list[str] = []
        parts.append(f"name={self.run_name}")
        if self.project:
            parts.append(f"project={_truncate(self.project)}")

        return f"{cls_name}({', '.join(parts)})"

    # endregion

    # region Seeding

    _rng: ClassVar[np.random.Generator | None] = None

    @staticmethod
    def generate_id(*, length: int = 8) -> str:
        """
        Generate a random ID of specified length.

        """
        if (rng := BaseConfig._rng) is None:
            rng = np.random.default_rng()

        alphabet = list(string.ascii_lowercase + string.digits)

        id = "".join(rng.choice(alphabet) for _ in range(length))
        return id

    @staticmethod
    def set_seed(seed: int | None = None) -> None:
        """
        Set the seed for the random number generator.

        Args:
            seed (int | None, optional): The seed value to set. If None, a seed based on the current time will be used. Defaults to None.

        Returns:
            None
        """
        if seed is None:
            seed = int(time.time() * 1000)
        log.critical(f"Seeding BaseConfig with seed {seed}")
        BaseConfig._rng = np.random.default_rng(seed)

    # endregion

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        hparams_key: str = "hyper_parameters",
    ):
        ckpt = torch.load(path)
        if (hparams := ckpt.get(hparams_key)) is None:
            raise ValueError(
                f"The checkpoint does not contain the `{hparams_key}` attribute. "
                "Are you sure this is a valid Lightning checkpoint?"
            )
        return cls.model_validate(hparams)

    def _nshtrainer_all_callback_configs(self) -> Iterable[CallbackConfigBase | None]:
        yield from self.trainer._nshtrainer_all_callback_configs()
