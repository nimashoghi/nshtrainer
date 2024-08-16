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


def find_symlinks(
    target_file: _Path,
    *search_directories: _Path,
    glob_pattern: str = "*",
):
    target_file = Path(target_file).resolve()
    symlinks: list[Path] = []

    for search_directory in search_directories:
        search_directory = Path(search_directory)
        for path in search_directory.rglob(glob_pattern):
            if path.is_symlink():
                try:
                    link_target = path.resolve()
                    if link_target.samefile(target_file):
                        symlinks.append(path)
                except FileNotFoundError:
                    # Handle broken symlinks
                    pass

    return symlinks
