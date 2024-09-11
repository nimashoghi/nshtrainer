from typing import Annotated, TypeAlias

import nshconfig as C

from .best_checkpoint import BestCheckpoint as BestCheckpoint
from .best_checkpoint import (
    BestCheckpointCallbackConfig as BestCheckpointCallbackConfig,
)
from .last_checkpoint import LastCheckpoint as LastCheckpoint
from .last_checkpoint import (
    LastCheckpointCallbackConfig as LastCheckpointCallbackConfig,
)
from .on_exception_checkpoint import OnExceptionCheckpoint as OnExceptionCheckpoint
from .on_exception_checkpoint import (
    OnExceptionCheckpointCallbackConfig as OnExceptionCheckpointCallbackConfig,
)

CheckpointCallbackConfig: TypeAlias = Annotated[
    BestCheckpointCallbackConfig
    | LastCheckpointCallbackConfig
    | OnExceptionCheckpointCallbackConfig,
    C.Field(discriminator="name"),
]
