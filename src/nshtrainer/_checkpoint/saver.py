import os
import shutil
from pathlib import Path

from lightning.pytorch import Trainer

from .metadata import _link_checkpoint_metadata, _remove_checkpoint_metadata


def _link_checkpoint(
    trainer: Trainer,
    filepath: str | Path | os.PathLike,
    linkpath: str | Path | os.PathLike,
    *,
    barrier: bool,
    metadata: bool,
):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if not isinstance(linkpath, Path):
        linkpath = Path(linkpath)

    if trainer.is_global_zero:
        if linkpath.exists():
            if linkpath.is_symlink() or linkpath.is_file():
                linkpath.unlink()
            elif linkpath.is_dir():
                shutil.rmtree(linkpath)
            _remove_checkpoint_metadata(linkpath)

        try:
            target_path = filepath.relative_to(linkpath.parent)
            linkpath.symlink_to(target_path)
        except OSError:
            # on Windows, special permissions are required to create symbolic links as a regular user
            # fall back to copying the file
            shutil.copy(filepath, linkpath)

        _link_checkpoint_metadata(filepath, linkpath)
    if barrier:
        trainer.strategy.barrier()


def _remove_checkpoint(
    trainer: Trainer,
    filepath: str | Path | os.PathLike,
    remove_metadata: bool = True,
):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    trainer.strategy.remove_checkpoint(filepath)
    _remove_checkpoint_metadata(filepath)
