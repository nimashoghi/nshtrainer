from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Literal, cast

import torch
import torch.distributed
from lightning.pytorch import LightningModule
from lightning.pytorch.profilers import PassThroughProfiler, Profiler
from typing_extensions import Never, TypeVar, override

from ..callbacks.rlp_sanity_checks import _RLPSanityCheckModuleMixin
from .config import BaseConfig
from .mixins.callback import CallbackModuleMixin
from .mixins.logger import LoggerLightningModuleMixin

log = logging.getLogger(__name__)

TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)


T = TypeVar("T", infer_variance=True)

ReduceOpStr = Literal[
    "avg",
    "mean",
    "band",
    "bor",
    "bxor",
    "max",
    "min",
    "premul_sum",
    "product",
    "sum",
]
VALID_REDUCE_OPS = (
    "avg",
    "mean",
    "band",
    "bor",
    "bxor",
    "max",
    "min",
    "premul_sum",
    "product",
    "sum",
)


class LightningModuleBase(
    _RLPSanityCheckModuleMixin,
    LoggerLightningModuleMixin,
    CallbackModuleMixin,
    LightningModule,
    ABC,
    Generic[TConfig],
):
    hparams: Never  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: Never  # pyright: ignore[reportIncompatibleMethodOverride]

    # region Config
    @torch.jit.unused
    @property
    def config(self) -> TConfig:
        return cast(TConfig, self.hparams)

    @property
    def debug(self) -> bool:
        if torch.jit.is_scripting():
            return False

        if (trainer := self._trainer) is None:
            return False

        from ..trainer import Trainer

        if not isinstance(trainer, Trainer):
            return False

        return trainer.debug

    @debug.setter
    def debug(self, value: bool):
        if torch.jit.is_scripting():
            return

        if (trainer := self._trainer) is None:
            return

        from ..trainer import Trainer

        if not isinstance(trainer, Trainer):
            return

        trainer.debug = value

    # endregion

    # region Debug

    @torch.jit.unused
    def breakpoint(self, rank_zero_only: bool = True):
        if (
            not rank_zero_only
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            breakpoint()

        if rank_zero_only and torch.distributed.is_initialized():
            _ = torch.distributed.barrier()

    @torch.jit.unused
    def ensure_finite(
        self,
        tensor: torch.Tensor,
        name: str | None = None,
        throw: bool = False,
    ):
        name_parts: list[str] = ["Tensor"]
        if name is not None:
            name_parts.append(name)
        name = " ".join(name_parts)

        not_finite = ~torch.isfinite(tensor)
        if not_finite.any():
            msg = f"{name} has {not_finite.sum().item()}/{not_finite.numel()} non-finite values."
            if throw:
                raise RuntimeError(msg)
            else:
                log.warning(msg)
            return False
        return True

    # endregion

    # region Profiler
    @property
    def profiler(self) -> Profiler:
        if (trainer := self._trainer) is None:
            raise RuntimeError("trainer is not defined")

        if not hasattr(trainer, "profiler"):
            raise RuntimeError("trainer does not have profiler")

        if (profiler := getattr(trainer, "profiler")) is None:
            profiler = PassThroughProfiler()

        return profiler

    # endregion

    # region Distributed
    def all_gather_object(
        self,
        object: T,
        group: torch.distributed.ProcessGroup | None = None,
    ) -> list[T]:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return [object]

        object_list = [cast(T, None) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(object_list, object, group=group)
        return object_list

    def barrier(self, name: str | None = None):
        return self.trainer.strategy.barrier(name=name)

    def reduce(
        self,
        tensor: torch.Tensor,
        reduce_op: torch.distributed.ReduceOp.RedOpType | ReduceOpStr,
        group: Any | None = None,
    ) -> torch.Tensor:
        if isinstance(reduce_op, str):
            # validate reduce_op
            if reduce_op not in VALID_REDUCE_OPS:
                raise ValueError(
                    f"reduce_op must be one of {VALID_REDUCE_OPS}, got {reduce_op}"
                )

        return self.trainer.strategy.reduce(tensor, group=group, reduce_op=reduce_op)

    # endregion

    # Our own custom __repr__ method.
    # Torch's __repr__ method is too verbose and doesn't provide any useful information.
    @override
    def __repr__(self):
        parts: list[str] = []
        parts.append(f"config={repr(self.config)}")
        parts.append(f"device={self.device}")
        if self.debug:
            parts.append("debug=True")

        parts_str = ", ".join(parts)
        return f"{self.__class__.__name__}({parts_str})"

    @classmethod
    @abstractmethod
    def config_cls(cls) -> type[TConfig]: ...

    @override
    def __init__(self, hparams: TConfig | Mapping[str, Any]):
        config_cls = self.config_cls()
        if not isinstance(hparams, config_cls):
            if not isinstance(hparams, Mapping):
                raise TypeError(
                    f"hparams must be a BaseConfig or a Mapping: {type(hparams)}"
                )
            hparams = config_cls.model_validate(hparams)
        hparams = config_cls.model_validate(hparams)

        super().__init__()
        self.save_hyperparameters(hparams)

    def zero_loss(self):
        """
        Returns a loss tensor with the value 0.
        It multiples each weight by 0 and returns the sum, so we don't run into issues with ununsed parameters in DDP.
        """
        loss = sum((0.0 * v).sum() for v in self.parameters() if v.requires_grad)
        loss = cast(torch.Tensor, loss)
        return loss
