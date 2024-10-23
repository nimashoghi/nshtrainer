from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from nshtrainer.model.config import BaseConfig as BaseConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "BaseConfig":
            return importlib.import_module("nshtrainer.model.config").BaseConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
