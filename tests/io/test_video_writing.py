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


def test_video_writer_build_output_params():
    """Test VideoWriter._build_output_params method."""
    # Default params (no keyframe_interval or no_audio)
    writer = VideoWriter("output.mp4", fps=30.0)
    params = writer.build_output_params()
    assert "-crf" in params
    assert "-preset" in params
    assert "-g" not in params
    assert "-an" not in params

    # With keyframe_interval
    writer = VideoWriter("output.mp4", fps=30.0, keyframe_interval=1.0)
    params = writer.build_output_params()
    gop_idx = params.index("-g")
    assert params[gop_idx + 1] == "30"  # 30 fps * 1.0 second

    # With keyframe_interval = 0.5 (15 frames at 30fps)
    writer = VideoWriter("output.mp4", fps=30.0, keyframe_interval=0.5)
    params = writer.build_output_params()
    gop_idx = params.index("-g")
    assert params[gop_idx + 1] == "15"

    # With no_audio
    writer = VideoWriter("output.mp4", fps=30.0, no_audio=True)
    params = writer.build_output_params()
    assert "-an" in params

    # With both
    writer = VideoWriter("output.mp4", fps=30.0, keyframe_interval=2.0, no_audio=True)
    params = writer.build_output_params()
    assert "-g" in params
    assert "-an" in params
    gop_idx = params.index("-g")
    assert params[gop_idx + 1] == "60"  # 30 fps * 2.0 seconds


def test_video_writer_with_keyframe_interval(centered_pair_low_quality_video, tmp_path):
    """Test VideoWriter with keyframe_interval parameter."""
    imgs = centered_pair_low_quality_video[:4]
    with VideoWriter(tmp_path / "output.mp4", keyframe_interval=0.5) as writer:
        for img in imgs:
            writer.write_frame(img)

    assert (tmp_path / "output.mp4").exists()
    vid = sio.load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)


def test_video_writer_with_no_audio(centered_pair_low_quality_video, tmp_path):
    """Test VideoWriter with no_audio parameter."""
    imgs = centered_pair_low_quality_video[:4]
    with VideoWriter(tmp_path / "output.mp4", no_audio=True) as writer:
        for img in imgs:
            writer.write_frame(img)

    assert (tmp_path / "output.mp4").exists()
    vid = sio.load_video(tmp_path / "output.mp4")
    assert vid.shape == (4, 384, 384, 1)
