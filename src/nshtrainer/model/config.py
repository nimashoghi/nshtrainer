import logging
from pathlib import Path

import nshconfig as C
import torch

log = logging.getLogger(__name__)


class BaseConfig(C.Config):
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        hparams_key: str = "hyper_parameters",
    ):
        ckpt = torch.load(path)
        if (hparams := ckpt.get(hparams_key)) is None:
            raise ValueError(
                f"The checkpoint does not contain the `{hparams_key}` attribute. "
                "Are you sure this is a valid Lightning checkpoint?"
            )
        return cls.model_validate(hparams)
