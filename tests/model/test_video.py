from multiprocessing import dummy
from statistics import median_grouped
from tokenize import single_quoted
from sleap_io.model.video import (
    Video,
)


def test_video_class():
    test_video = Video(filename="123.mp4", shape=(1, 2, 3, 4), backend=None)
    assert test_video.filename == "123.mp4"
    assert test_video.shape == (1, 2, 3, 4)
    assert Video.fixup_path("123.mp4") == "123.mp4"
