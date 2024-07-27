import getpass
import inspect
import logging
import os
import platform
import socket
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import nshconfig as C
from typing_extensions import TypeVar

from ..util.slurm import parse_slurm_node_list

if TYPE_CHECKING:
    from .base import LightningModuleBase
    from .config import BaseConfig


log = logging.getLogger(__name__)

T = TypeVar("T", infer_variance=True)


def _try_get(fn: Callable[[], T | None]) -> T | None:
    try:
        return fn()
    except Exception as e:
        log.warning(f"Failed to get value: {e}")
        return None


class EnvironmentClassInformationConfig(C.Config):
    """Configuration for class information in the environment."""

    name: str | None
    """The name of the class."""

    module: str | None
    """The module where the class is defined."""

    full_name: str | None
    """The fully qualified name of the class."""

    file_path: Path | None
    """The file path where the class is defined."""

    source_file_path: Path | None
    """The source file path of the class, if available."""

    @classmethod
    def empty(cls):
        return cls(
            name=None,
            module=None,
            full_name=None,
            file_path=None,
            source_file_path=None,
        )

    @classmethod
    def from_class(cls, cls_: type):
        name = cls_.__name__
        module = cls_.__module__
        full_name = f"{cls_.__module__}.{cls_.__qualname__}"

        file_path = inspect.getfile(cls_)
        source_file_path = inspect.getsourcefile(cls_)
        return cls(
            name=name,
            module=module,
            full_name=full_name,
            file_path=Path(file_path),
            source_file_path=Path(source_file_path) if source_file_path else None,
        )

    @classmethod
    def from_instance(cls, instance: object):
        return cls.from_class(type(instance))


class EnvironmentSLURMInformationConfig(C.Config):
    """Configuration for SLURM environment information."""

    hostname: str | None
    """The hostname of the current node."""

    hostnames: list[str] | None
    """List of hostnames for all nodes in the job."""

    job_id: str | None
    """The SLURM job ID."""

    raw_job_id: str | None
    """The raw SLURM job ID."""

    array_job_id: str | None
    """The SLURM array job ID, if applicable."""

    array_task_id: str | None
    """The SLURM array task ID, if applicable."""

    num_tasks: int | None
    """The number of tasks in the SLURM job."""

    num_nodes: int | None
    """The number of nodes in the SLURM job."""

    node: str | int | None
    """The node ID or name."""

    global_rank: int | None
    """The global rank of the current process."""

    local_rank: int | None
    """The local rank of the current process within its node."""

    @classmethod
    def empty(cls):
        return cls(
            hostname=None,
            hostnames=None,
            job_id=None,
            raw_job_id=None,
            array_job_id=None,
            array_task_id=None,
            num_tasks=None,
            num_nodes=None,
            node=None,
            global_rank=None,
            local_rank=None,
        )

    @classmethod
    def from_current_environment(cls):
        try:
            from lightning.fabric.plugins.environments.slurm import SLURMEnvironment

            if not SLURMEnvironment.detect():
                return None

            hostname = socket.gethostname()
            hostnames = [hostname]
            if node_list := os.environ.get("SLURM_JOB_NODELIST", ""):
                hostnames = parse_slurm_node_list(node_list)

            raw_job_id = os.environ["SLURM_JOB_ID"]
            job_id = raw_job_id
            array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
            array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if array_job_id and array_task_id:
                job_id = f"{array_job_id}_{array_task_id}"

            num_tasks = int(os.environ["SLURM_NTASKS"])
            num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])

            node_id = os.environ.get("SLURM_NODEID")

            global_rank = int(os.environ["SLURM_PROCID"])
            local_rank = int(os.environ["SLURM_LOCALID"])

            return cls(
                hostname=hostname,
                hostnames=hostnames,
                job_id=job_id,
                raw_job_id=raw_job_id,
                array_job_id=array_job_id,
                array_task_id=array_task_id,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                node=node_id,
                global_rank=global_rank,
                local_rank=local_rank,
            )
        except (ImportError, RuntimeError, ValueError, KeyError):
            return None


class EnvironmentLSFInformationConfig(C.Config):
    """Configuration for LSF environment information."""

    hostname: str | None
    """The hostname of the current node."""

    hostnames: list[str] | None
    """List of hostnames for all nodes in the job."""

    job_id: str | None
    """The LSF job ID."""

    array_job_id: str | None
    """The LSF array job ID, if applicable."""

    array_task_id: str | None
    """The LSF array task ID, if applicable."""

    num_tasks: int | None
    """The number of tasks in the LSF job."""

    num_nodes: int | None
    """The number of nodes in the LSF job."""

    node: str | int | None
    """The node ID or name."""

    global_rank: int | None
    """The global rank of the current process."""

    local_rank: int | None
    """The local rank of the current process within its node."""

    @classmethod
    def empty(cls):
        return cls(
            hostname=None,
            hostnames=None,
            job_id=None,
            array_job_id=None,
            array_task_id=None,
            num_tasks=None,
            num_nodes=None,
            node=None,
            global_rank=None,
            local_rank=None,
        )

    @classmethod
    def from_current_environment(cls):
        try:
            import os
            import socket

            hostname = socket.gethostname()
            hostnames = [hostname]
            if node_list := os.environ.get("LSB_HOSTS", ""):
                hostnames = node_list.split()

            job_id = os.environ["LSB_JOBID"]
            array_job_id = os.environ.get("LSB_JOBINDEX")
            array_task_id = os.environ.get("LSB_JOBINDEX")

            num_tasks = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
            num_nodes = len(set(hostnames))

            node_id = (
                os.environ.get("LSB_HOSTS", "").split().index(hostname)
                if "LSB_HOSTS" in os.environ
                else None
            )

            # LSF doesn't have direct equivalents for global_rank and local_rank
            # You might need to calculate these based on your specific setup
            global_rank = int(os.environ.get("PMI_RANK", 0))
            local_rank = int(os.environ.get("LSB_RANK", 0))

            return cls(
                hostname=hostname,
                hostnames=hostnames,
                job_id=job_id,
                array_job_id=array_job_id,
                array_task_id=array_task_id,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                node=node_id,
                global_rank=global_rank,
                local_rank=local_rank,
            )
        except (ImportError, RuntimeError, ValueError, KeyError):
            return None


def _psutil():
    import psutil

    return psutil


class EnvironmentLinuxEnvironmentConfig(C.Config):
    """Configuration for Linux environment information."""

    user: str | None
    """The current user."""

    hostname: str | None
    """The hostname of the machine."""

    system: str | None
    """The operating system name."""

    release: str | None
    """The operating system release."""

    version: str | None
    """The operating system version."""

    machine: str | None
    """The machine type."""

    processor: str | None
    """The processor type."""

    cpu_count: int | None
    """The number of CPUs."""

    memory: int | None
    """The total system memory in bytes."""

    uptime: timedelta | None
    """The system uptime."""

    boot_time: float | None
    """The system boot time as a timestamp."""

    load_avg: tuple[float, float, float] | None
    """The system load average (1, 5, and 15 minutes)."""

    @classmethod
    def empty(cls):
        return cls(
            user=None,
            hostname=None,
            system=None,
            release=None,
            version=None,
            machine=None,
            processor=None,
            cpu_count=None,
            memory=None,
            uptime=None,
            boot_time=None,
            load_avg=None,
        )

    @classmethod
    def from_current_environment(cls):
        return cls(
            user=_try_get(lambda: getpass.getuser()),
            hostname=_try_get(lambda: platform.node()),
            system=_try_get(lambda: platform.system()),
            release=_try_get(lambda: platform.release()),
            version=_try_get(lambda: platform.version()),
            machine=_try_get(lambda: platform.machine()),
            processor=_try_get(lambda: platform.processor()),
            cpu_count=_try_get(lambda: os.cpu_count()),
            memory=_try_get(lambda: _psutil().virtual_memory().total),
            uptime=_try_get(lambda: timedelta(seconds=_psutil().boot_time())),
            boot_time=_try_get(lambda: _psutil().boot_time()),
            load_avg=_try_get(lambda: os.getloadavg()),
        )


class EnvironmentSnapshotConfig(C.Config):
    """Configuration for environment snapshot information."""

    snapshot_dir: Path | None
    """The directory where the snapshot is stored."""

    modules: list[str] | None
    """List of modules included in the snapshot."""

    @classmethod
    def empty(cls):
        return cls(snapshot_dir=None, modules=None)

    @classmethod
    def from_current_environment(cls):
        draft = cls.draft()
        if snapshot_dir := os.environ.get("NSHRUNNER_SNAPSHOT_DIR"):
            draft.snapshot_dir = Path(snapshot_dir)
        if modules := os.environ.get("NSHRUNNER_SNAPSHOT_MODULES"):
            draft.modules = modules.split(",")
        return draft.finalize()


class EnvironmentConfig(C.Config):
    """Configuration for the overall environment."""

    cwd: Path | None
    """The current working directory."""

    snapshot: EnvironmentSnapshotConfig | None
    """The environment snapshot configuration."""

    python_executable: Path | None
    """The path to the Python executable."""

    python_path: list[Path] | None
    """The Python path."""

    python_version: str | None
    """The Python version."""

    config: EnvironmentClassInformationConfig | None
    """The configuration class information."""

    model: EnvironmentClassInformationConfig | None
    """The model class information."""

    data: EnvironmentClassInformationConfig | None
    """The data class information."""

    linux: EnvironmentLinuxEnvironmentConfig | None
    """The Linux environment information."""

    slurm: EnvironmentSLURMInformationConfig | None
    """The SLURM environment information."""

    lsf: EnvironmentLSFInformationConfig | None
    """The LSF environment information."""

    base_dir: Path | None
    """The base directory for the run."""

    log_dir: Path | None
    """The directory for logs."""

    checkpoint_dir: Path | None
    """The directory for checkpoints."""

    stdio_dir: Path | None
    """The directory for standard input/output files."""

    seed: int | None
    """The global random seed."""

    seed_workers: bool | None
    """Whether to seed workers."""

    @classmethod
    def empty(cls):
        return cls(
            cwd=None,
            snapshot=None,
            python_executable=None,
            python_path=None,
            python_version=None,
            config=None,
            model=None,
            data=None,
            linux=None,
            slurm=None,
            lsf=None,
            base_dir=None,
            log_dir=None,
            checkpoint_dir=None,
            stdio_dir=None,
            seed=None,
            seed_workers=None,
        )

    @classmethod
    def from_current_environment(
        cls,
        model: "LightningModuleBase",
        root_config: "BaseConfig",
    ):
        draft = cls.draft()
        draft.cwd = Path(os.getcwd())
        draft.python_executable = Path(sys.executable)
        draft.python_path = [Path(path) for path in sys.path]
        draft.python_version = sys.version
        draft.config = EnvironmentClassInformationConfig.from_instance(root_config)
        draft.model = EnvironmentClassInformationConfig.from_instance(model)
        draft.slurm = EnvironmentSLURMInformationConfig.from_current_environment()
        draft.lsf = EnvironmentLSFInformationConfig.from_current_environment()
        draft.base_dir = root_config.directory.resolve_run_root_directory(
            root_config.id
        )
        draft.log_dir = root_config.directory.resolve_subdirectory(
            root_config.id, "log"
        )
        draft.checkpoint_dir = root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )
        draft.stdio_dir = root_config.directory.resolve_subdirectory(
            root_config.id, "stdio"
        )
        draft.seed = (
            int(seed_str) if (seed_str := os.environ.get("PL_GLOBAL_SEED")) else None
        )
        draft.seed_workers = (
            bool(int(seed_everything))
            if (seed_everything := os.environ.get("PL_SEED_WORKERS"))
            else None
        )
        draft.linux = EnvironmentLinuxEnvironmentConfig.from_current_environment()
        draft.snapshot = EnvironmentSnapshotConfig.from_current_environment()
        return draft.finalize()
