import pytest

from sleap_io.model.video import (
    Video,
)


class DummyVideo:
    """Fake video backend,returns frames with all zeros.

    This can be useful when you want to look at labels for a dataset but don't
    have access to the real video.
    """

    filename: str = ""
    height: int = 2000
    width: int = 2000
    frames: int = 10000
    channels: int = 1
    dummy: bool = True


@pytest.fixture
def test_video():
    return DummyVideo()