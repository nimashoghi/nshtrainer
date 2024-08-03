import os
import shutil
from pathlib import Path

from lightning.pytorch import Trainer

from .metadata import _link_checkpoint_metadata, _remove_checkpoint_metadata


def _link_checkpoint(
    filepath: str | Path | os.PathLike,
    linkpath: str | Path | os.PathLike,
    *,
    metadata: bool,
    remove_existing: bool = True,
):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if not isinstance(linkpath, Path):
        linkpath = Path(linkpath)

    if remove_existing:
        if linkpath.exists():
            if linkpath.is_symlink() or linkpath.is_file():
                linkpath.unlink()
            elif linkpath.is_dir():
                shutil.rmtree(linkpath)

        if metadata:
            _remove_checkpoint_metadata(linkpath)

    try:
        linkpath.symlink_to(filepath.relative_to(linkpath.parent))
    except OSError:
        # on Windows, special permissions are required to create symbolic links as a regular user
        # fall back to copying the file
        shutil.copy(filepath, linkpath)

    if metadata:
        _link_checkpoint_metadata(filepath, linkpath)


def _remove_checkpoint(
    trainer: Trainer,
    filepath: str | Path | os.PathLike,
    *,
    metadata: bool,
):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    trainer.strategy.remove_checkpoint(filepath)
    if metadata:
        _remove_checkpoint_metadata(filepath)
