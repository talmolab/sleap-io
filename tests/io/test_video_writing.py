"""Tests for the sleap_io.io.video_writing module."""

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


def test_mjpeg_writer_write_frames(centered_pair_low_quality_video, tmp_path):
    """Test MJPEGFrameWriter.write_frames method."""
    imgs = centered_pair_low_quality_video[:3]

    with MJPEGFrameWriter(tmp_path / "test_frames.avi") as writer:
        writer.write_frames(imgs)

    assert (tmp_path / "test_frames.avi").exists()
    vid = sio.load_video(tmp_path / "test_frames.avi")
    assert vid.shape == (3, 384, 384, 1)
