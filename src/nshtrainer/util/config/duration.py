import math
from typing import Annotated, Literal

import nshconfig as C


class Steps(C.Config):
    kind: Literal["steps"] = "steps"

    value: Annotated[int, C.Field(ge=0)]
    """Number of steps."""

    def to_steps(self, steps_per_epoch: int):
        return self


class Epochs(C.Config):
    kind: Literal["epochs"] = "epochs"

    value: Annotated[int | float, C.Field(ge=0)]
    """Number of epochs."""

    def to_steps(self, steps_per_epoch: int):
        value = self.value * steps_per_epoch
        if not isinstance(value, int):
            value = int(math.ceil(value))

        return Steps(value=value)


Duration = Annotated[Steps | Epochs, C.Field(discriminator="kind")]