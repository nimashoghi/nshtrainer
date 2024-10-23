from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, cast

import torch
from lightning.pytorch import LightningDataModule
from typing_extensions import Never, TypeVar

from ..model.mixins.callback import CallbackRegistrarModuleMixin

TConfig = TypeVar("TConfig", infer_variance=True)


class LightningDataModuleBase(
    CallbackRegistrarModuleMixin,
    LightningDataModule,
    ABC,
    Generic[TConfig],
):
    hparams: Never  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: Never  # pyright: ignore[reportIncompatibleMethodOverride]

    @classmethod
    @abstractmethod
    def config_cls(cls) -> type[TConfig]: ...

    @torch.jit.unused
    @property
    def config(self) -> TConfig:
        return cast(TConfig, self.hparams)
