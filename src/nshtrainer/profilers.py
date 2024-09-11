import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import nshconfig as C
from lightning.pytorch.profilers import Profiler
from typing_extensions import override

if TYPE_CHECKING:
    from .trainer.config import TrainerConfig

log = logging.getLogger(__name__)


class BaseProfilerConfig(C.Config, ABC):
    dirpath: Path | None = None
    """
    Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
        ``trainer.log_dir`` (from :class:`~lightning.pytorch.loggers.tensorboard.TensorBoardLogger`)
        will be used.
    """
    filename: str | None = None
    """
    If present, filename where the profiler results will be saved instead of printing to stdout.
        The ``.txt`` extension will be used automatically.
    """

    @abstractmethod
    def create_profiler(self, trainer_config: "TrainerConfig") -> Profiler: ...


class SimpleProfilerConfig(BaseProfilerConfig):
    name: Literal["simple"] = "simple"

    extended: bool = True
    """
    If ``True``, adds extra columns representing number of calls and percentage of
        total time spent onrespective action.
    """

    @override
    def create_profiler(self, trainer_config):
        from lightning.pytorch.profilers.simple import SimpleProfiler

        dirpath = trainer_config.dir_or_default_subdir(self.dirpath, "profile")

        if (filename := self.filename) is None:
            filename = f"{trainer_config.run.id}_profile.txt"

        return SimpleProfiler(
            extended=self.extended,
            dirpath=dirpath,
            filename=filename,
        )


class AdvancedProfilerConfig(BaseProfilerConfig):
    name: Literal["advanced"] = "advanced"

    line_count_restriction: float = 1.0
    """
    This can be used to limit the number of functions
        reported for each action. either an integer (to select a count of lines),
        or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
    """

    @override
    def create_profiler(self, trainer_config):
        from lightning.pytorch.profilers.advanced import AdvancedProfiler

        dirpath = trainer_config.dir_or_default_subdir(self.dirpath, "profile")

        if (filename := self.filename) is None:
            filename = f"{trainer_config.run.id}_profile.txt"

        return AdvancedProfiler(
            line_count_restriction=self.line_count_restriction,
            dirpath=dirpath,
            filename=filename,
        )


class PyTorchProfilerConfig(BaseProfilerConfig):
    name: Literal["pytorch"] = "pytorch"

    group_by_input_shapes: bool = False
    """Include operator input shapes and group calls by shape."""

    emit_nvtx: bool = False
    """
    Context manager that makes every autograd operation emit an NVTX range
        Run::

            nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

        To visualize, you can either use::

            nvvp trace_name.prof
            torch.autograd.profiler.load_nvprof(path)
    """

    export_to_chrome: bool = True
    """
    Whether to export the sequence of profiled operators for Chrome.
        It will generate a ``.json`` file which can be read by Chrome.
    """

    row_limit: int = 20
    """
    Limit the number of rows in a table, ``-1`` is a special value that
        removes the limit completely.
    """

    sort_by_key: str | None = None
    """
    Attribute used to sort entries. By default
        they are printed in the same order as they were registered.
        Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
        ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
        ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
    """

    record_module_names: bool = True
    """Whether to add module names while recording autograd operation."""

    table_kwargs: dict[str, Any] | None = None
    """Dictionary with keyword arguments for the summary table."""

    additional_profiler_kwargs: dict[str, Any] = {}
    """Keyword arguments for the PyTorch profiler. This depends on your PyTorch version"""

    @override
    def create_profiler(self, trainer_config):
        from lightning.pytorch.profilers.pytorch import PyTorchProfiler

        dirpath = trainer_config.dir_or_default_subdir(self.dirpath, "profile")

        if (filename := self.filename) is None:
            filename = f"{trainer_config.run.id}_profile.txt"

        return PyTorchProfiler(
            group_by_input_shapes=self.group_by_input_shapes,
            emit_nvtx=self.emit_nvtx,
            export_to_chrome=self.export_to_chrome,
            row_limit=self.row_limit,
            sort_by_key=self.sort_by_key,
            record_module_names=self.record_module_names,
            table_kwargs=self.table_kwargs,
            dirpath=dirpath,
            filename=filename,
            **self.additional_profiler_kwargs,
        )


ProfilerConfig: TypeAlias = Annotated[
    SimpleProfilerConfig | AdvancedProfilerConfig | PyTorchProfilerConfig,
    C.Field(discriminator="name"),
]
