"""Tests for video reference restoration control when saving labels."""

import pytest
import json
import h5py
from pathlib import Path
import numpy as np

from sleap_io import Labels, Video, load_file, save_file
from sleap_io.io.slp import write_videos, read_videos, write_labels, video_to_dict
from sleap_io.io.video_reading import HDF5Video, MediaVideo


def test_labels_save_restore_original_videos_api(tmp_path, slp_minimal_pkg):
    """Test the restore_original_videos parameter in Labels.save()."""
    # Load a .pkg.slp file
    labels = load_file(slp_minimal_pkg)

    # Verify it has embedded videos with source_video metadata
    assert len(labels.videos) == 1
    video = labels.videos[0]
    assert isinstance(video.backend, HDF5Video)
    assert video.backend.has_embedded_images
    assert video.source_video is not None
    original_video_path = video.source_video.filename

    # Test default behavior (restore_original_videos=True)
    output_default = tmp_path / "test_default.slp"
    labels.save(output_default, embed=False)

    # Load and check that original video is restored
    labels_default = load_file(output_default)
    assert labels_default.videos[0].filename == original_video_path
    assert labels_default.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test new behavior (restore_original_videos=False)
    output_preserve = tmp_path / "test_preserve_source.slp"
    labels.save(output_preserve, embed=False, restore_original_videos=False)

    # Load and check that source .pkg.slp is referenced
    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"
    assert labels_preserve.videos[0].backend_metadata["has_embedded_images"] == True


def test_save_slp_restore_original_videos_api(tmp_path, slp_minimal_pkg):
    """Test the restore_original_videos parameter in save_slp()."""
    from sleap_io.io.main import save_slp

    # Load a .pkg.slp file
    labels = load_file(slp_minimal_pkg)

    # Test default behavior
    output_default = tmp_path / "test_default_api.slp"
    save_slp(labels, output_default, embed=False)

    labels_default = load_file(output_default)
    assert labels_default.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test new behavior
    output_preserve = tmp_path / "test_preserve_api.slp"
    save_slp(labels, output_preserve, embed=False, restore_original_videos=False)

    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"


def test_video_metadata_preservation(tmp_path, slp_minimal_pkg):
    """Test that video metadata is preserved correctly in all modes."""
    # Load fresh labels for metadata extraction
    labels = load_file(slp_minimal_pkg)
    video = labels.videos[0]

    # Store original metadata for comparison
    # For minimal_instance.pkg.slp, source_video IS the original video
    original_backend_metadata = video.source_video.backend_metadata.copy()
    source_backend_metadata = video.backend_metadata.copy()

    # Test EMBED mode
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_embed = tmp_path / "test_embed.slp"
    labels.save(output_embed, embed=True)

    with h5py.File(output_embed, "r") as f:
        # Check that both original and source video metadata are stored
        assert "video0" in f
        # Since source_video IS the original, it should be stored as original_video
        assert "original_video" in f["video0"]
        assert "source_video" in f["video0"]

        # Verify original video metadata (should be the MediaVideo)
        assert isinstance(f["video0/original_video"], h5py.Group)
        original_json = json.loads(f["video0/original_video"].attrs["json"])
        assert original_json["backend"]["type"] == "MediaVideo"
        assert (
            original_json["backend"]["filename"]
            == original_backend_metadata["filename"]
        )

        # Verify source video metadata (should be the .pkg.slp file)
        assert isinstance(f["video0/source_video"], h5py.Group)
        source_json = json.loads(f["video0/source_video"].attrs["json"])
        assert source_json["backend"]["type"] == "HDF5Video"
        assert source_json["backend"]["filename"] == slp_minimal_pkg

    # Test RESTORE_ORIGINAL mode (default)
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_restore = tmp_path / "test_restore.slp"
    labels.save(output_restore, embed=False, restore_original_videos=True)

    # Load and verify the video reference is restored to original
    labels_restore = load_file(output_restore)
    assert labels_restore.videos[0].filename == original_backend_metadata["filename"]
    assert labels_restore.videos[0].backend_metadata["type"] == "MediaVideo"

    # Test PRESERVE_SOURCE mode (new)
    labels = load_file(slp_minimal_pkg)  # Fresh load
    output_preserve = tmp_path / "test_preserve.slp"
    labels.save(output_preserve, embed=False, restore_original_videos=False)

    # Load and verify the video reference is preserved to source .pkg.slp
    labels_preserve = load_file(output_preserve)
    assert labels_preserve.videos[0].filename == slp_minimal_pkg
    assert labels_preserve.videos[0].backend_metadata["type"] == "HDF5Video"


def test_multiple_save_load_cycles(tmp_path, slp_minimal_pkg):
    """Test that video lineage is preserved through multiple save/load cycles."""
    # First cycle: Load .pkg.slp and save with preserve_source
    labels1 = load_file(slp_minimal_pkg)
    original_video_path = labels1.videos[0].source_video.filename

    output1 = tmp_path / "cycle1.slp"
    labels1.save(output1, embed=False, restore_original_videos=False)

    # Second cycle: Load cycle1.slp and save again
    labels2 = load_file(output1)
    assert labels2.videos[0].filename == slp_minimal_pkg
    # For minimal_instance.pkg.slp, source_video IS the original video
    # In PRESERVE_SOURCE mode, the original video metadata should be preserved
    assert labels2.videos[0].source_video is not None
    assert labels2.videos[0].source_video.filename == original_video_path

    output2 = tmp_path / "cycle2.slp"
    labels2.save(output2, embed=False, restore_original_videos=False)

    # Third cycle: Verify metadata is still preserved
    labels3 = load_file(output2)
    # In PRESERVE_SOURCE mode, it should still reference the original .pkg.slp
    assert labels3.videos[0].filename == slp_minimal_pkg
    # Verify metadata persistence through multiple cycles
    assert labels3.videos[0].source_video is not None
    assert labels3.videos[0].source_video.filename == original_video_path


def test_unavailable_video_handling(tmp_path):
    """Test handling of videos when files are not available."""
    # Create a Labels object with a video that doesn't exist
    fake_video = Video(
        filename="/nonexistent/original.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "shape": [100, 384, 384, 3],
            "filename": "/nonexistent/original.mp4",
            "grayscale": False,
            "bgr": True,
            "dataset": "",
            "input_format": "",
        },
    )

    labels = Labels(videos=[fake_video])

    # Save and verify metadata is preserved
    output = tmp_path / "test_unavailable.slp"
    labels.save(output, embed=False)

    # Load and check metadata was preserved
    loaded = load_file(output)
    assert loaded.videos[0].filename == "/nonexistent/original.mp4"
    assert loaded.videos[0].backend is None or not loaded.videos[0].exists()
    assert loaded.videos[0].backend_metadata["type"] == "MediaVideo"
    assert loaded.videos[0].backend_metadata["shape"] == [100, 384, 384, 3]


def test_video_reference_mode_enum():
    """Test that VideoReferenceMode enum is properly defined."""
    # This will fail until the enum is implemented
    from sleap_io.io.slp import VideoReferenceMode

    assert VideoReferenceMode.EMBED.value == "embed"
    assert VideoReferenceMode.RESTORE_ORIGINAL.value == "restore_original"
    assert VideoReferenceMode.PRESERVE_SOURCE.value == "preserve_source"


def test_write_videos_with_reference_mode(tmp_path, slp_minimal_pkg):
    """Test the internal write_videos function with VideoReferenceMode."""
    from sleap_io.io.slp import VideoReferenceMode

    labels = load_file(slp_minimal_pkg)
    videos = labels.videos

    # Test PRESERVE_SOURCE mode
    output = tmp_path / "test_internal.slp"
    write_videos(output, videos, reference_mode=VideoReferenceMode.PRESERVE_SOURCE)

    # Read back and verify
    loaded_videos = read_videos(output)
    assert loaded_videos[0].filename == slp_minimal_pkg
    assert loaded_videos[0].backend_metadata["type"] == "HDF5Video"


def test_backwards_compatibility(tmp_path, slp_minimal_pkg):
    """Test that the default behavior maintains backwards compatibility."""
    labels = load_file(slp_minimal_pkg)
    original_video_path = labels.videos[0].source_video.filename

    # Default behavior should restore original videos
    output = tmp_path / "test_compat.slp"
    labels.save(output, embed=False)  # No restore_original_videos parameter

    loaded = load_file(output)
    assert loaded.videos[0].filename == original_video_path
    assert loaded.videos[0].backend_metadata["type"] == "MediaVideo"


def test_video_to_dict_with_none_backend(tmp_path):
    """Test that video_to_dict handles videos with backend=None correctly."""
    video = Video(
        filename="test.mp4",
        backend=None,
        backend_metadata={
            "type": "MediaVideo",
            "filename": "test.mp4",
            "shape": [10, 100, 100, 1],
            "grayscale": True,
        },
    )

    video_dict = video_to_dict(video, labels_path=tmp_path / "test.slp")

    assert video_dict["filename"] == "test.mp4"
    assert video_dict["backend"] == video.backend_metadata


def test_video_original_video_field(slp_minimal_pkg):
    """Test that Video objects have the new original_video field."""
    labels = load_file(slp_minimal_pkg)
    video = labels.videos[0]

    # Current implementation has source_video, not original_video
    # This test will fail until we implement the field rename
    assert hasattr(video, "original_video")
    assert video.original_video is None  # Not set for current files

    # TODO: The source_video should become original_video when the field rename is complete
    assert video.source_video is not None  # This is current behavior


def test_complex_workflow(tmp_path, slp_minimal_pkg):
    """Test a complex workflow with training and inference results."""
    # Load training data
    train_labels = load_file(slp_minimal_pkg)

    # Simulate saving for distribution (embed=True)
    train_pkg = tmp_path / "train.pkg.slp"
    train_labels.save(train_pkg, embed=True)

    # Load in inference environment
    inference_labels = load_file(train_pkg)

    # Simulate predictions (in practice would come from a model)
    predictions = Labels(
        videos=inference_labels.videos,
        skeletons=inference_labels.skeletons,
        labeled_frames=[],  # Would contain predicted instances
    )

    # Save predictions referencing the training package
    predictions_output = tmp_path / "predictions_on_train.slp"
    predictions.save(predictions_output, embed=False, restore_original_videos=False)

    # Load predictions and verify they reference train.pkg.slp
    loaded_predictions = load_file(predictions_output)
    assert loaded_predictions.videos[0].filename == train_pkg.as_posix()
    assert loaded_predictions.videos[0].backend_metadata["type"] == "HDF5Video"
    assert loaded_predictions.videos[0].backend_metadata["has_embedded_images"] == True

    # Verify metadata preservation through the workflow
    # The video objects from inference_labels already have source_video metadata
    # which is preserved when we create the predictions Labels object
    assert loaded_predictions.videos[0].source_video is not None

    # The source_video should point to minimal_instance.pkg.slp (the original training data)
    # This is correct because we're using the same video objects from inference_labels
    assert loaded_predictions.videos[0].source_video.filename == slp_minimal_pkg
    assert (
        loaded_predictions.videos[0].source_video.backend_metadata["type"]
        == "HDF5Video"
    )

    # And that should have the original MediaVideo as its source
    assert loaded_predictions.videos[0].source_video.source_video is not None
    assert (
        loaded_predictions.videos[0].source_video.source_video.backend_metadata["type"]
        == "MediaVideo"
    )


def test_write_videos_backwards_compatibility():
    """Test backwards compatibility with restore_source parameter."""
    from sleap_io.io.slp import write_videos, VideoReferenceMode
    from sleap_io.model.video import Video
    import tempfile

    video = Video(
        filename="test.mp4",
        backend_metadata={
            "type": "MediaVideo",
            "shape": [1, 100, 100, 1],
            "filename": "test.mp4",
            "grayscale": True,
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test.slp"

        # Test restore_source=True with reference_mode=None (should use RESTORE_ORIGINAL)
        write_videos(str(output), [video], restore_source=True, reference_mode=None)

        # Test restore_source=False with reference_mode=None (should use EMBED)
        write_videos(str(output), [video], restore_source=False, reference_mode=None)


def test_video_lineage_edge_cases():
    """Test edge cases in video lineage metadata handling."""
    from sleap_io.io.slp import write_videos, VideoReferenceMode
    from sleap_io.model.video import Video
    import tempfile

    # Test case 1: Video with original_video already set
    original = Video(
        filename="original.mp4",
        backend_metadata={
            "type": "MediaVideo",
            "shape": [10, 100, 100, 1],
            "filename": "original.mp4",
            "grayscale": True,
        },
    )

    video_with_original = Video(
        filename="current.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=None,
        original_video=original,  # This should be saved
    )

    # Test case 2: source_video has original_video
    source_with_original = Video(
        filename="source.pkg.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "source.pkg.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        original_video=original,
    )

    video_with_source_original = Video(
        filename="current2.slp",
        backend_metadata={
            "type": "HDF5Video",
            "shape": [10, 100, 100, 1],
            "filename": "current2.slp",
            "dataset": "video0/video",
            "has_embedded_images": True,
            "grayscale": True,
        },
        source_video=source_with_original,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test_lineage.slp"

        # Write videos with different lineage scenarios
        write_videos(
            str(output),
            [video_with_original, video_with_source_original],
            reference_mode=VideoReferenceMode.EMBED,
            original_videos=[video_with_original, video_with_source_original],
        )
