from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pytest

from nshtrainer.util.path import try_symlink_or_copy


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def source_file(temp_dir):
    """Create a source file for testing."""
    file_path = temp_dir / "source_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture
def source_dir(temp_dir):
    """Create a source directory with files for testing."""
    dir_path = temp_dir / "source_dir"
    dir_path.mkdir()
    (dir_path / "file1.txt").write_text("test content 1")
    (dir_path / "file2.txt").write_text("test content 2")
    return dir_path


def test_try_symlink_or_copy_new_file(source_file, temp_dir):
    """Test creating a new symlink/copy when target doesn't exist."""
    link_path = temp_dir / "link_file.txt"

    # Execute the function
    result = try_symlink_or_copy(source_file, link_path)

    # Verify the result
    assert result is True
    assert link_path.exists()
    assert link_path.read_text() == "test content"

    # Check if it's a symlink on Unix and a copy on Windows
    if platform.system() != "Windows":
        assert link_path.is_symlink()


def test_try_symlink_or_copy_existing_file(source_file, temp_dir):
    """Test overwriting an existing file."""
    link_path = temp_dir / "existing_file.txt"
    link_path.write_text("old content")

    # Execute the function
    result = try_symlink_or_copy(source_file, link_path)

    # Verify the result
    assert result is True
    assert link_path.exists()
    assert link_path.read_text() == "test content"


def test_try_symlink_or_copy_directory(source_dir, temp_dir):
    """Test symlink/copy with a directory."""
    link_path = temp_dir / "link_dir"

    # Execute the function
    result = try_symlink_or_copy(source_dir, link_path, target_is_directory=True)

    # Verify the result
    assert result is True
    assert link_path.exists()
    assert (link_path / "file1.txt").exists()
    assert (link_path / "file1.txt").read_text() == "test content 1"
    assert (link_path / "file2.txt").exists()
    assert (link_path / "file2.txt").read_text() == "test content 2"


def test_try_symlink_or_copy_no_remove_existing(source_file, temp_dir):
    """Test when remove_existing=False and target exists."""
    link_path = temp_dir / "dont_remove.txt"
    link_path.write_text("old content")

    # Execute the function with remove_existing=False
    result = try_symlink_or_copy(source_file, link_path, remove_existing=False)

    # On Windows, it will still overwrite the file
    # On Unix, it will fail because the file exists
    if platform.system() == "Windows":
        assert result is True
        assert link_path.read_text() == "test content"
    else:
        assert result is False
        assert link_path.read_text() == "old content"


def test_try_symlink_or_copy_nonexistent_source(temp_dir):
    """Test with a nonexistent source file."""
    source_path = temp_dir / "nonexistent_file.txt"
    link_path = temp_dir / "link_to_nowhere.txt"

    # Execute the function
    with pytest.raises(FileNotFoundError):
        try_symlink_or_copy(source_path, link_path, throw_on_invalid_target=True)


def test_try_symlink_or_copy_absolute_path(source_file, temp_dir):
    """Test creating a symlink with absolute paths."""
    link_path = temp_dir / "absolute_link.txt"

    # Execute the function with relative=False
    result = try_symlink_or_copy(source_file, link_path, relative=False)

    # Verify the result
    assert result is True
    assert link_path.exists()
    assert link_path.read_text() == "test content"

    # If on Unix, check if it's using an absolute path
    if platform.system() != "Windows" and link_path.is_symlink():
        target = os.readlink(link_path)
        assert os.path.isabs(target) or target == source_file.name
