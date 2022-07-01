<<<<<<< HEAD
from sleap_io.model.video import (
    Video,
)
=======
from sleap_io import Video
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472


def test_video_class():
    test_video = Video(filename="123.mp4", shape=(1, 2, 3, 4), backend=None)
    assert test_video.filename == "123.mp4"
    assert test_video.shape == (1, 2, 3, 4)
