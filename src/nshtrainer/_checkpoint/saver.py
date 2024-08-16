import logging
import os
import shutil
from pathlib import Path

from lightning.pytorch import Trainer

from ..util.path import get_relative_path
from .metadata import _link_checkpoint_metadata, _remove_checkpoint_metadata

log = logging.getLogger(__name__)


def _link_checkpoint(
    filepath: str | Path | os.PathLike,
    linkpath: str | Path | os.PathLike,
    *,
    metadata: bool,
    remove_existing: bool = True,
):
    filepath = Path(filepath)
    linkpath = Path(linkpath)

    if remove_existing:
        try:
            if linkpath.exists():
                if linkpath.is_dir():
                    shutil.rmtree(linkpath, ignore_errors=True)
                else:
                    linkpath.unlink(missing_ok=True)
        except Exception:
            log.exception(f"Failed to remove {linkpath}")

        if metadata:
            _remove_checkpoint_metadata(linkpath)

    try:
        linkpath.symlink_to(get_relative_path(linkpath, filepath))
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
    filepath = Path(filepath)

    trainer.strategy.remove_checkpoint(filepath)

    if metadata:
        _remove_checkpoint_metadata(filepath)
