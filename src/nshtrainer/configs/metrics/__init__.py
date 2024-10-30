from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshtrainer.metrics import MetricConfig as MetricConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "MetricConfig":
            return importlib.import_module("nshtrainer.metrics").MetricConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import _config as _config