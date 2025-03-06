from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nshtrainer._checkpoint.saver import link_checkpoint, remove_checkpoint


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def source_checkpoint(temp_dir):
    """Create a source checkpoint file for testing."""
    checkpoint_path = temp_dir / "model.ckpt"
    checkpoint_path.write_text("checkpoint content")
    return checkpoint_path


@pytest.fixture
def metadata_file(temp_dir, source_checkpoint):
    """Create a metadata file for the checkpoint."""
    metadata_path = source_checkpoint.with_suffix(".metadata.json")
    metadata_path.write_text('{"test": "metadata"}')
    return metadata_path


def test_link_checkpoint_with_metadata(source_checkpoint, metadata_file, temp_dir):
    """Test linking a checkpoint with metadata=True."""
    link_path = temp_dir / "linked_model.ckpt"

    # Execute the function
    link_checkpoint(source_checkpoint, link_path, metadata=True)

    # Verify the checkpoint was linked
    assert link_path.exists()
    assert link_path.read_text() == "checkpoint content"

    # Verify the metadata was also linked
    metadata_link = link_path.with_suffix(".metadata.json")
    assert metadata_link.exists()
    assert metadata_link.read_text() == '{"test": "metadata"}'


def test_link_checkpoint_without_metadata(source_checkpoint, metadata_file, temp_dir):
    """Test linking a checkpoint with metadata=False."""
    link_path = temp_dir / "linked_model_no_metadata.ckpt"

    # Execute the function
    link_checkpoint(source_checkpoint, link_path, metadata=False)

    # Verify the checkpoint was linked
    assert link_path.exists()
    assert link_path.read_text() == "checkpoint content"

    # Verify no metadata file was linked
    metadata_link = link_path.with_suffix(".metadata.json")
    assert not metadata_link.exists()


def test_link_checkpoint_existing_file(source_checkpoint, metadata_file, temp_dir):
    """Test overwriting an existing checkpoint link."""
    # Create existing files
    link_path = temp_dir / "existing_link.ckpt"
    link_path.write_text("old checkpoint")
    metadata_link = link_path.with_suffix(".metadata.json")
    metadata_link.write_text('{"old": "metadata"}')

    # Execute the function
    link_checkpoint(source_checkpoint, link_path, metadata=True)

    # Verify the files were overwritten
    assert link_path.exists()
    assert link_path.read_text() == "checkpoint content"
    assert metadata_link.exists()
    assert metadata_link.read_text() == '{"test": "metadata"}'


def test_link_checkpoint_no_remove_existing(source_checkpoint, metadata_file, temp_dir):
    """Test with remove_existing=False."""
    # Create existing files
    link_path = temp_dir / "dont_remove.ckpt"
    link_path.write_text("old checkpoint")
    metadata_link = link_path.with_suffix(".metadata.json")
    metadata_link.write_text('{"old": "metadata"}')

    # Execute the function with remove_existing=False
    link_checkpoint(source_checkpoint, link_path, metadata=True, remove_existing=False)

    # The behavior here is platform-dependent, but on any platform the function should complete
    assert link_path.exists()
    assert metadata_link.exists()


def test_link_checkpoint_nonexistent_metadata(source_checkpoint, temp_dir):
    """Test linking a checkpoint with no metadata file."""
    link_path = temp_dir / "no_metadata_link.ckpt"

    # Execute the function - this should not raise an error even though source has no metadata
    link_checkpoint(source_checkpoint, link_path, metadata=False)

    # Verify the checkpoint was linked
    assert link_path.exists()
    assert link_path.read_text() == "checkpoint content"

    # No metadata file should exist
    metadata_link = link_path.with_suffix(".metadata.json")
    assert not metadata_link.exists()


def test_link_checkpoint_nonexistent_metadata(source_checkpoint, temp_dir):
    """Test linking a checkpoint with no metadata file."""
    link_path = temp_dir / "no_metadata_link.ckpt"

    # Execute the function - this SHOULD raise an error because metadata=True
    with pytest.raises(FileNotFoundError):
        link_checkpoint(source_checkpoint, link_path, metadata=True)


def test_remove_checkpoint(temp_dir):
    """Test removing a checkpoint."""
    # Create a checkpoint file
    checkpoint_path = temp_dir / "to_remove.ckpt"
    checkpoint_path.write_text("checkpoint to remove")
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata_path.write_text('{"to": "remove"}')

    # Create a mock trainer
    mock_trainer = MagicMock()

    # Execute the function
    remove_checkpoint(mock_trainer, checkpoint_path, metadata=True)

    # Verify the trainer's remove_checkpoint was called
    mock_trainer.strategy.remove_checkpoint.assert_called_once_with(checkpoint_path)

    # The metadata file should be gone (the actual checkpoint removal is mocked)
    assert not metadata_path.exists()


def test_remove_checkpoint_without_metadata(temp_dir):
    """Test removing a checkpoint without touching metadata."""
    # Create a checkpoint file
    checkpoint_path = temp_dir / "keep_metadata.ckpt"
    checkpoint_path.write_text("checkpoint content")
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata_path.write_text('{"keep": "this"}')

    # Create a mock trainer
    mock_trainer = MagicMock()

    # Execute the function
    remove_checkpoint(mock_trainer, checkpoint_path, metadata=False)

    # Verify the trainer's remove_checkpoint was called
    mock_trainer.strategy.remove_checkpoint.assert_called_once_with(checkpoint_path)

    # The metadata file should still exist
    assert metadata_path.exists()
    assert metadata_path.read_text() == '{"keep": "this"}'
