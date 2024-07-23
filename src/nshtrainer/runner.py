from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, cast

from nshrunner import Config, RunInfo, RunnerConfigDict
from nshrunner import Runner as _Runner
from typing_extensions import TypeVar, TypeVarTuple, Unpack, override

from .model.config import BaseConfig

TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TArguments = TypeVarTuple("TArguments")
TReturn = TypeVar("TReturn", infer_variance=True)


@dataclass(frozen=True)
class Runner(
    _Runner[Unpack[tuple[TConfig, Unpack[TArguments]]], TReturn],
    Generic[TConfig, Unpack[TArguments], TReturn],
):
    @override
    def default_validate_fn(self, config: TConfig, *args: Unpack[TArguments]) -> None:
        super().default_validate_fn(config, *args)

    @override
    def default_info_fn(self, config: TConfig, *args: Unpack[TArguments]) -> RunInfo:
        run_info = super().default_info_fn(config, *args)
        return {
            **run_info,
            "id": config.id,
            "base_dir": config.directory.project_root,
        }


def runner(
    run_fn: Callable[[TConfig, Unpack[TArguments]], TReturn],
    info_fn: Callable[[TConfig, Unpack[TArguments]], RunInfo] | None = None,
    validate_fn: Callable[
        [TConfig, Unpack[TArguments]], tuple[TConfig, Unpack[TArguments]] | None
    ]
    | None = None,
    transform_fns: list[
        Callable[[TConfig, Unpack[TArguments]], tuple[TConfig, Unpack[TArguments]]]
    ] = [],
    **config: Unpack[RunnerConfigDict],
):
    return Runner(
        config=Config(**cast(Any, config)),
        run_fn=run_fn,
        info_fn=info_fn,
        validate_fn=validate_fn,
        transform_fns=transform_fns,
    )
