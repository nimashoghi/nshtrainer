from __future__ import annotations

import logging
import string
import time
from typing import TYPE_CHECKING, Annotated, ClassVar, Generic

import nshconfig as C
import numpy as np
from typing_extensions import TypeVar

from ._directory import DirectoryConfig
from .model.config import BaseConfig
from .trainer._config import TrainerConfig
from .util._environment_info import EnvironmentConfig

log = logging.getLogger(__name__)

TTrainerConfig = TypeVar(
    "TTrainerConfig",
    bound=TrainerConfig,
    infer_variance=True,
    default=TrainerConfig,
)
TModelConfig = TypeVar(
    "TModelConfig",
    bound=BaseConfig,
    infer_variance=True,
    default=BaseConfig,
)
TDataModuleConfig = TypeVar(
    "TDataModuleConfig",
    bound=C.Config,
    infer_variance=True,
    default=C.Config,
)


class RunConfig(C.Config, Generic[TTrainerConfig, TModelConfig, TDataModuleConfig]):
    """
    Configuration for an active instance of a run.
    """

    id: str = C.Field(default_factory=lambda: RunConfig.generate_id())
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

    # region Seeding

    _rng: ClassVar[np.random.Generator | None] = None

    @classmethod
    def generate_id(cls, *, length: int = 8) -> str:
        """
        Generate a random ID of specified length.

        """
        if (rng := cls._rng) is None:
            rng = np.random.default_rng()

        alphabet = list(string.ascii_lowercase + string.digits)

        id = "".join(rng.choice(alphabet) for _ in range(length))
        return id

    @classmethod
    def set_seed(cls, seed: int | None = None) -> None:
        """
        Set the seed for the random number generator.

        Args:
            seed (int | None, optional): The seed value to set. If None, a seed based on the current time will be used. Defaults to None.

        Returns:
            None
        """
        if seed is None:
            seed = int(time.time() * 1000)
        log.critical(f"Seeding RunConfig with seed {seed}")
        RunConfig._rng = np.random.default_rng(seed)

    # endregion
