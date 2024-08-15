import os
from pathlib import Path
from typing import TypeAlias

_Path: TypeAlias = str | Path | os.PathLike


def get_relative_path(source: _Path, destination: _Path):
    # Get the absolute paths
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)

    # Split the paths into components
    source_parts = source.split(os.sep)
    destination_parts = destination.split(os.sep)

    # Find the point where the paths diverge
    i = 0
    for i in range(min(len(source_parts), len(destination_parts))):
        if source_parts[i] != destination_parts[i]:
            break
    else:
        i += 1

    # Build the relative path
    up = os.sep.join([".." for _ in range(len(source_parts) - i - 1)])
    down = os.sep.join(destination_parts[i:])

    return Path(os.path.normpath(os.path.join(up, down)))
