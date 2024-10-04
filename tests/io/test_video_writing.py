"""Tests for the sleap_io.io.video_writing module."""

import sleap_io as sio
from sleap_io.io.video_writing import VideoWriter


def test_video_writer(centered_pair_low_quality_video, tmp_path):
    imgs = centered_pair_low_quality_video[:4]
    with VideoWriter(tmp_path / "output.mp4") as writer:
        for img in imgs:
            writer.write_frame(img)

    assert (tmp_path / "output.mp4").exists()
    vid = sio.load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)
