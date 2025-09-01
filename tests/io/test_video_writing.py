"""Tests for the sleap_io.io.video_writing module."""

import numpy as np

import sleap_io as sio
from sleap_io.io.video_writing import MJPEGFrameWriter, VideoWriter


def test_video_writer(centered_pair_low_quality_video, tmp_path):
    imgs = centered_pair_low_quality_video[:4]
    with VideoWriter(tmp_path / "output.mp4") as writer:
        for img in imgs:
            writer.write_frame(img)

    assert (tmp_path / "output.mp4").exists()
    vid = sio.load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)


def test_mjpeg_frame_writer_write_frames(tmp_path):
    """Test MJPEGFrameWriter.write_frames method."""
    # Create test frames (use dimensions divisible by 16 to avoid resizing)
    frames = [
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)
    ]

    # Test write_frames method
    output_path = tmp_path / "test_mjpeg.avi"
    with MJPEGFrameWriter(output_path) as writer:
        writer.write_frames(frames)

    # Verify the video was created
    assert output_path.exists()

    # Load and verify the video
    vid = sio.load_video(str(output_path))
    assert vid.shape[0] == 5  # 5 frames
    assert vid.shape[1:3] == (128, 128)  # height, width


def test_mjpeg_ffmpeg_version_handling(tmp_path, monkeypatch):
    """Test MJPEG writer handles ffmpeg version detection errors."""
    # This tests lines 177-179 in video_writing.py
    import imageio_ffmpeg

    # Test with version parsing error
    def bad_get_version():
        return "invalid.version.string"

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_version", bad_get_version)

    output_path = tmp_path / "test_version_error.avi"

    # Create writer with frame_durations to trigger version check
    writer = MJPEGFrameWriter(
        str(output_path),
        frame_durations=[0.033] * 5,  # This triggers version detection
    )

    # Should not crash despite version parsing error
    frames = [
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)
    ]
    writer.write_frames(frames)

    assert output_path.exists()

    # Test with AttributeError (missing get_ffmpeg_version)
    del imageio_ffmpeg.get_ffmpeg_version

    output_path2 = tmp_path / "test_attr_error.avi"
    writer2 = MJPEGFrameWriter(str(output_path2), frame_durations=[0.033] * 5)

    # Should not crash despite missing function
    writer2.write_frames(frames)
    assert output_path2.exists()
