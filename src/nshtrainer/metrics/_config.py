from typing import Literal

import nshconfig as C


class MetricConfig(C.Config):
    name: str
    """The name of the primary metric."""

    mode: Literal["min", "max"]
    """
    The mode of the primary metric:
    - "min" for metrics that should be minimized (e.g., loss)
    - "max" for metrics that should be maximized (e.g., accuracy)
    """

    @property
    def validation_monitor(self) -> str:
        return f"val/{self.name}"

    def __post_init__(self):
        for split in ("train", "val", "test", "predict"):
            if self.name.startswith(f"{split}/"):
                raise ValueError(
                    f"Primary metric name should not start with '{split}/'. "
                    f"Just use '{self.name[len(split) + 1:]}' instead. "
                    "The split name is automatically added depending on the context."
                )

    @classmethod
    def loss(cls, mode: Literal["min", "max"] = "min"):
        return cls(name="loss", mode=mode)