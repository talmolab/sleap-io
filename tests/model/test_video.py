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
    hdf5_ = HDF5Video()
    numpy_ = NumpyVideo()
    median_ = MediaVideo()
    imgstore_ = ImgStoreVideo()
    single_ = SingleImageVideo()

    # Dummy

    assert dummy_.filename == ""
    assert dummy_.height == 2000
    assert dummy_.width == 2000
    assert dummy_.frames == 10000
    assert dummy_.channels == 1
    assert dummy_.dummy == True

    # Hdf5

    assert hdf5_.filename == None
    assert hdf5_.dataset == None
    assert hdf5_.input_format == "channels_last"
    assert hdf5_.convert_range == True

    # Numpy

    # Imgstore

    assert imgstore_.filename == None
    assert imgstore_.index_by_original == True
    assert imgstore_._store_ == None
    assert imgstore_._img_ == None

    # SingleImageVideo

    assert single_.filenamestr == None
    assert type(single_.filenameslist) == list
    assert single_.height_int == None
    assert single_.width_int == None
    assert single_.channels_int == None

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
