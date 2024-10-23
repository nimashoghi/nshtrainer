from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshtrainer.model import BaseConfig as BaseConfig
    from nshtrainer.model import DirectoryConfig as DirectoryConfig
    from nshtrainer.model import MetricConfig as MetricConfig
    from nshtrainer.model import TrainerConfig as TrainerConfig
    from nshtrainer.model.config import CallbackConfigBase as CallbackConfigBase
    from nshtrainer.model.config import EnvironmentConfig as EnvironmentConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "BaseConfig":
            return importlib.import_module("nshtrainer.model").BaseConfig
        if name == "CallbackConfigBase":
            return importlib.import_module("nshtrainer.model.config").CallbackConfigBase
        if name == "DirectoryConfig":
            return importlib.import_module("nshtrainer.model").DirectoryConfig
        if name == "EnvironmentConfig":
            return importlib.import_module("nshtrainer.model.config").EnvironmentConfig
        if name == "MetricConfig":
            return importlib.import_module("nshtrainer.model").MetricConfig
        if name == "TrainerConfig":
            return importlib.import_module("nshtrainer.model").TrainerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import base as base
from . import config as config
