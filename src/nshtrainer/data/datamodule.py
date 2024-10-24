from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, cast

import nshconfig as C
from lightning.pytorch import LightningDataModule
from typing_extensions import Never, TypeVar, deprecated, override

from ..model.mixins.callback import CallbackRegistrarModuleMixin
from ..model.mixins.debug import _DebugModuleMixin

THparams = TypeVar("THparams", bound=C.Config, infer_variance=True)


class LightningDataModuleBase(
    _DebugModuleMixin,
    CallbackRegistrarModuleMixin,
    LightningDataModule,
    ABC,
    Generic[THparams],
):
    @property
    @override
    def hparams(self) -> THparams:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(THparams, super().hparams)

    @property
    @override
    def hparams_initial(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        hparams = cast(THparams, super().hparams_initial)
        return cast(Never, {"datamodule": hparams.model_dump(mode="json")})

    @property
    @deprecated("Use `hparams` instead")
    def config(self):
        return cast(Never, self.hparams)

    @classmethod
    @abstractmethod
    def hparams_cls(cls) -> type[THparams]: ...

    @override
    def __init__(self, hparams: THparams | Mapping[str, Any]):
        super().__init__()

        # Validate and save hyperparameters
        hparams_cls = self.hparams_cls()
        if isinstance(hparams, Mapping):
            hparams = hparams_cls.model_validate(hparams)
        elif not isinstance(hparams, hparams_cls):
            raise TypeError(
                f"Expected hparams to be either a Mapping or an instance of {hparams_cls}, got {type(hparams)}"
            )
        hparams = hparams.model_deep_validate()
        self.save_hyperparameters(hparams)
