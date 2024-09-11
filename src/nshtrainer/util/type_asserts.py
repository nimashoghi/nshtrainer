from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer as LightningTrainer
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from ..model import BaseConfig, LightningModuleBase
    from ..trainer import Trainer


def is_nshtrainer(trainer: LightningTrainer) -> "TypeIs[Trainer]":
    from ..trainer import Trainer

    return isinstance(trainer, Trainer)


def is_nshmodule(module: LightningModule) -> "TypeIs[LightningModuleBase[BaseConfig]]":
    from ..model import LightningModuleBase

    return isinstance(module, LightningModuleBase)
