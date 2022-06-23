from multiprocessing import dummy
from statistics import median_grouped
from tokenize import single_quoted
from sleap_io.model.video import (
    HDF5Video,
    NumpyVideo,
    MediaVideo,
    ImgStoreVideo,
    SingleImageVideo,
    DummyVideo,
    Video,
)


def test_defaults():

    dummy_ = DummyVideo()
    media_ = MediaVideo()
    imgstore_ = ImgStoreVideo()
    single_ = SingleImageVideo()

    # Dummy

    assert dummy_.filename == ""
    assert dummy_.height == 2000
    assert dummy_.width == 2000
    assert dummy_.frames == 10000
    assert dummy_.channels == 1
    assert dummy_.dummy == True

    # Imgstore

    assert imgstore_.filename == None
    assert imgstore_.index_by_original == True
    assert imgstore_._store_ == None
    assert imgstore_._img_ == None

    # SingleImageVideo

    assert single_.filename == None
    assert type(single_.filenames) == list
    assert single_.height_ == None
    assert single_.width_ == None
    assert single_.channels_ == None

    # class: Video

    # Confirms that properties of dummy video still hold
    # when a video class is created from dummy video

    video_dummy = Video(dummy_)
    assert video_dummy.num_frames == 10000
    assert video_dummy.shape == (
        dummy_.frames,
        dummy_.height,
        dummy_.width,
        dummy_.channels,
    )
