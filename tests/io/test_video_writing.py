"""Tests for the sleap_io.io.video_writing module."""

import numpy as np

import sleap_io as sio
from sleap_io.io.video_writing import (
    MJPEGFrameWriter,
    VideoWriter,
    _pad_to_macro_block,
)


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


def test_pad_to_macro_block_divisible():
    """Test _pad_to_macro_block with already-divisible dimensions."""
    # 512x512 is divisible by 16, no padding needed
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    frame[100, 200, :] = [255, 0, 0]

    padded = _pad_to_macro_block(frame)

    assert padded.shape == (512, 512, 3)
    assert np.array_equal(padded, frame)  # Should be unchanged


def test_pad_to_macro_block_height_not_divisible():
    """Test _pad_to_macro_block when height needs padding."""
    # 406 % 16 = 6, needs 10 px padding
    frame = np.zeros((406, 720, 3), dtype=np.uint8)
    frame[0, 0, :] = [255, 0, 0]  # Marker at origin
    frame[405, 719, :] = [0, 255, 0]  # Marker at last pixel

    padded = _pad_to_macro_block(frame)

    assert padded.shape == (416, 720, 3)  # Height padded to 416
    assert np.array_equal(padded[0, 0, :], [255, 0, 0])  # Origin preserved
    assert np.array_equal(padded[405, 719, :], [0, 255, 0])  # Last pixel preserved
    assert np.all(padded[406:, :, :] == 0)  # Padding is black


def test_pad_to_macro_block_width_not_divisible():
    """Test _pad_to_macro_block when width needs padding."""
    # 406 % 16 = 6, needs 10 px padding
    frame = np.zeros((720, 406, 3), dtype=np.uint8)
    frame[0, 0, :] = [255, 0, 0]
    frame[719, 405, :] = [0, 255, 0]

    padded = _pad_to_macro_block(frame)

    assert padded.shape == (720, 416, 3)  # Width padded to 416
    assert np.array_equal(padded[0, 0, :], [255, 0, 0])
    assert np.array_equal(padded[719, 405, :], [0, 255, 0])
    assert np.all(padded[:, 406:, :] == 0)  # Padding is black


def test_pad_to_macro_block_both_dimensions():
    """Test _pad_to_macro_block when both dimensions need padding."""
    # 400 % 16 = 0 (no padding), 300 % 16 = 12 (needs 4 px)
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    padded = _pad_to_macro_block(frame)

    assert padded.shape == (400, 304, 3)


def test_pad_to_macro_block_grayscale():
    """Test _pad_to_macro_block with 2D grayscale frame."""
    frame = np.zeros((406, 720), dtype=np.uint8)
    frame[0, 0] = 255  # Marker at origin

    padded = _pad_to_macro_block(frame)

    assert padded.shape == (416, 720)  # 2D output
    assert padded[0, 0] == 255  # Origin preserved
    assert np.all(padded[406:, :] == 0)  # Padding is black


def test_pad_to_macro_block_custom_block_size():
    """Test _pad_to_macro_block with custom macro_block_size."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # With macro_block_size=8
    padded = _pad_to_macro_block(frame, macro_block_size=8)
    assert padded.shape == (104, 104, 3)  # 100 -> 104

    # With macro_block_size=32
    padded = _pad_to_macro_block(frame, macro_block_size=32)
    assert padded.shape == (128, 128, 3)  # 100 -> 128


def test_video_writer_preserves_coordinates(tmp_path):
    """Test that VideoWriter preserves keypoint coordinates with non-divisible dims."""
    # Create a frame with dimensions NOT divisible by 16
    height, width = 406, 720

    # Create frame with colored rows at specific positions
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, :, 0] = 255  # Row 0: red channel
    frame[200, :, 1] = 255  # Row 200: green channel
    frame[405, :, 2] = 255  # Row 405: blue channel (last row)

    # Write video
    output_path = tmp_path / "test_coords.mp4"
    with VideoWriter(output_path, fps=30) as writer:
        for _ in range(5):
            writer.write_frame(frame)

    # Load and verify
    import sleap_io as sio

    vid = sio.load_video(output_path)

    # Output should have padded dimensions
    assert vid.shape == (5, 416, 720, 3)

    # Extract first frame and check coordinates
    out_frame = vid[0]

    # Check that markers are at the correct rows (with tolerance for compression)
    # Row 0 should have highest red channel value in top rows
    assert out_frame[0, 360, 0] > out_frame[10, 360, 0]
    # Row 200 should have highest green channel value around that row
    assert out_frame[200, 360, 1] > out_frame[100, 360, 1]
    # Row 405 should have highest blue channel value in bottom non-padded rows
    assert out_frame[405, 360, 2] > out_frame[350, 360, 2]

    # Verify that padding area (rows 406-415) is mostly black
    padding_region = out_frame[406:, :, :]
    assert padding_region.mean() < 50  # Mostly black (some compression artifacts)


def test_video_writer_call_method(tmp_path):
    """Test VideoWriter.__call__ method."""
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    output_path = tmp_path / "test_call.mp4"

    with VideoWriter(output_path, fps=30) as writer:
        # Use __call__ method instead of write_frame
        writer(frame)
        writer(frame)

    import sleap_io as sio

    vid = sio.load_video(output_path)
    assert vid.shape[0] == 2  # Two frames
