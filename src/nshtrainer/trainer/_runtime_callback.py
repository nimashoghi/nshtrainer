import datetime
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from lightning.pytorch.callbacks.callback import Callback
from typing_extensions import override

log = logging.getLogger(__name__)


Stage: TypeAlias = Literal["train", "validate", "test", "predict"]
ALL_STAGES = ("train", "validate", "test", "predict")


class RuntimeTrackerCallback(Callback):
    def __init__(self):
        super().__init__()
        self._start_time: dict[Stage, float] = {}
        self._end_time: dict[Stage, float] = {}
        self._offsets = {stage: 0.0 for stage in ALL_STAGES}

    def start_time(self, stage: Stage) -> float | None:
        """Return the start time of a particular stage (in seconds)"""
        return self._start_time[stage]

    def end_time(self, stage: Stage) -> float | None:
        """Return the end time of a particular stage (in seconds)"""
        return self._end_time[stage]

    def time_elapsed(self, stage: Stage) -> float:
        """Return the time elapsed for a particular stage (in seconds)"""
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offsets[stage]
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    @override
    def on_train_start(self, trainer, pl_module):
        self._start_time["train"] = time.monotonic()

    @override
    def on_train_end(self, trainer, pl_module):
        self._end_time["train"] = time.monotonic()

    @override
    def on_validation_start(self, trainer, pl_module):
        self._start_time["validate"] = time.monotonic()

    @override
    def on_validation_end(self, trainer, pl_module):
        self._end_time["validate"] = time.monotonic()

    @override
    def on_test_start(self, trainer, pl_module):
        self._start_time["test"] = time.monotonic()

    @override
    def on_test_end(self, trainer, pl_module):
        self._end_time["test"] = time.monotonic()

    @override
    def on_predict_start(self, trainer, pl_module):
        self._start_time["predict"] = time.monotonic()

    @override
    def on_predict_end(self, trainer, pl_module):
        self._end_time["predict"] = time.monotonic()

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "time_elapsed": {stage: self.time_elapsed(stage) for stage in ALL_STAGES}
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]):
        time_elapsed: dict[Stage, float] = state_dict.get("time_elapsed", {})
        for stage in ALL_STAGES:
            self._offsets[stage] = time_elapsed.get(stage, 0)
