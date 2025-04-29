from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, cast, overload

import torch.nn as nn
from typing_extensions import TypeVar, override

TModule = TypeVar("TModule", bound=nn.Module, infer_variance=True)


class TypedModuleList(nn.ModuleList, Generic[TModule]):
    def __init__(self, modules: Iterable[TModule] | None = None) -> None:
        super().__init__(modules)

    @overload
    def __getitem__(self, idx: slice) -> TypedModuleList[TModule]: ...

    @overload
    def __getitem__(self, idx: int) -> TModule: ...

    @override
    def __getitem__(self, idx: int | slice) -> TModule | TypedModuleList[TModule]:
        return cast(TModule | TypedModuleList[TModule], super().__getitem__(idx))

    @override
    def __setitem__(self, idx: int, module: TModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().__setitem__(idx, module)

    @override
    def __iter__(self) -> Iterator[TModule]:
        return cast(Iterator[TModule], super().__iter__())

    @override
    def __iadd__(self, modules: Iterable[TModule]) -> TypedModuleList[TModule]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(TypedModuleList[TModule], super().__iadd__(modules))

    @override
    def __add__(self, modules: Iterable[TModule]) -> TypedModuleList[TModule]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(TypedModuleList[TModule], super().__add__(modules))

    @override
    def insert(self, idx: int, module: TModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().insert(idx, module)

    @override
    def append(self, module: TModule) -> TypedModuleList[TModule]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(TypedModuleList[TModule], super().append(module))

    @override
    def extend(self, modules: Iterable[TModule]) -> TypedModuleList[TModule]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(TypedModuleList[TModule], super().extend(modules))
