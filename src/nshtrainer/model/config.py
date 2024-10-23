from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import nshconfig as C

log = logging.getLogger(__name__)


class BaseConfig(C.Config):
    if not TYPE_CHECKING:
        trainer_config: Any | None = None
        datamodule_config: Any | None = None
