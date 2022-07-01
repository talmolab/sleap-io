from sleap_io import Video


def test_video_class():
    test_video = Video(filename="123.mp4", shape=(1, 2, 3, 4), backend=None)
    assert test_video.filename == "123.mp4"
    assert test_video.shape == (1, 2, 3, 4)
