from __future__ import annotations

from typing import Literal

from typing_extensions import override

from ._base import BaseLoggerConfig


class CSVLoggerConfig(BaseLoggerConfig):
    name: Literal["csv"] = "csv"

    enabled: bool = True
    """Enable CSV logging."""

    priority: int = 0
    """Priority of the logger. Higher priority loggers are created first."""

    prefix: str = ""
    """A string to put at the beginning of metric keys."""

    flush_logs_every_n_steps: int = 100
    """How often to flush logs to disk."""

    @override
    def create_logger(self, root_config):
        if not self.enabled:
            return None

        from lightning.pytorch.loggers.csv_logs import CSVLogger

        save_dir = root_config.directory._resolve_log_directory_for_logger(
            root_config.id,
            self,
        )
        return CSVLogger(
            save_dir=save_dir,
            name=root_config.run_name,
            version=root_config.id,
            prefix=self.prefix,
            flush_logs_every_n_steps=self.flush_logs_every_n_steps,
        )
