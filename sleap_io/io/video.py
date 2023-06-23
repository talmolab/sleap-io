"""Backends for reading and writing videos."""

from __future__ import annotations

import sys
from io import BytesIO
from typing import Optional, Tuple, Union
import json
import attrs
import h5py
import imageio.v3 as iio
import numpy as np

try:
    import cv2
except ImportError:
    pass

try:
    import imageio_ffmpeg
except ImportError:
    pass

try:
    import av
except ImportError:
    pass


def _get_valid_kwargs(cls, kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in cls.__attrs_attrs__}


@attrs.define
class VideoBackend:
    filename: str
    grayscale: Optional[bool] = None
    _cached_shape: Optional[Tuple[int, int, int, int]] = None

    @classmethod
    def from_filename(
        cls,
        filename: str,
        dataset: Optional[str] = None,
        grayscale: Optional[bool] = None,
        **kwargs,
    ) -> VideoBackend:
        """Create a VideoBackend from a filename.

        Args:
            filename: Path to video file.
            dataset: Name of dataset in HDF5 file.
            grayscale: Whether to force grayscale. If None, autodetect on first frame
                load.

        Returns:
            VideoBackend subclass instance.
        """
        if filename.endswith(MediaVideo.EXTS):
            return MediaVideo(
                filename, grayscale=grayscale, **_get_valid_kwargs(MediaVideo, kwargs)
            )
        elif filename.endswith(HDF5Video.EXTS):
            valid_kwargs = {
                k: v for k, v in kwargs.items() if k in MediaVideo.__attrs_attrs__
            }
            return HDF5Video(
                filename,
                dataset=dataset,
                grayscale=grayscale,
                **_get_valid_kwargs(HDF5Video, kwargs),
            )
        else:
            raise ValueError(f"Unknown video file type: {filename}")

    def read_frame(self, frame_idx: int) -> np.ndarray:
        raise NotImplementedError

    def read_frames(self, frame_inds: list) -> np.ndarray:
        return np.stack([self.read_frame(i) for i in frame_inds], axis=0)

    def read_test_frame(self) -> np.ndarray:
        return self.read_frame(0)

    def detect_grayscale(self) -> bool:
        img = self.read_test_frame()
        is_grayscale = bool(np.alltrue(img[..., 0] == img[..., -1]))
        self.grayscale = is_grayscale
        return is_grayscale

    @property
    def num_frames(self) -> int:
        raise NotImplementedError

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return self.get_frame(0).shape

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        if self._cached_shape is not None:
            return self._cached_shape
        else:
            shape = (self.num_frames,) + self.img_shape
            self._cached_shape = shape
            return shape

    @property
    def frames(self) -> int:
        return self.shape[0]

    def __len__(self) -> int:
        return self.shape[0]

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if self.grayscale is None:
            self.detect_grayscale()

        img = self.read_frame(frame_idx)

        if self.grayscale:
            img = img[..., [0]]

        return img

    def get_frames(self, frame_inds: list) -> np.ndarray:
        if self.grayscale is None:
            self.detect_grayscale()

        imgs = self.read_frames(frame_inds)

        if self.grayscale:
            imgs = imgs[..., [0]]

        return imgs

    def __getitem__(self, ind: Union[int, list[int]]) -> np.ndarray:
        if np.isscalar(ind):
            return self.get_frame(ind)
        else:
            return self.get_frames(ind)


@attrs.define
class MediaVideo(VideoBackend):
    plugin: str = attrs.field(
        validator=attrs.validators.in_(["pyav", "FFMPEG", "opencv"])
    )

    EXTS = ("mp4", "avi", "mov", "mj2", "mkv")

    @plugin.default
    def _default_plugin(self) -> str:
        if "cv2" in sys.modules:
            return "opencv"
        elif "imageio_ffmpeg" in sys.modules:
            return "FFMPEG"
        elif "av" in sys.modules:
            return "pyav"
        else:
            raise ImportError(
                "No video plugins found. Install opencv-python, imageio-ffmpeg, or av."
            )

    @property
    def num_frames(self) -> int:
        if self.plugin == "opencv":
            return cv2.VideoCapture(self.filename).get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            return iio.improps(self.filename, plugin="pyav").shape[0]

    def read_frame(self, frame_idx: int) -> np.ndarray:
        if self.plugin == "opencv":
            reader = cv2.VideoCapture(self.filename)
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, img = reader.read()
        else:
            with iio.imopen(self.filename, "r", plugin=self.plugin) as vid:
                img = vid.read(index=frame_idx)
        return img

    def read_frames(self, frame_inds: list) -> np.ndarray:
        if self.plugin == "opencv":
            reader = cv2.VideoCapture(self.filename)
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_inds[0])
            imgs = []
            for idx in frame_inds:
                if reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, img = reader.read()
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)

        else:
            with iio.imopen(self.filename, "r", plugin=self.plugin) as vid:
                imgs = np.stack([vid.read(idx) for idx in frame_inds], axis=0)
        return imgs


@attrs.define
class HDF5Video(VideoBackend):
    dataset: Optional[str] = None
    input_format: str = attrs.field(
        default="channels_last",
        validator=attrs.validators.in_(["channels_last", "channels_first"]),
    )
    frame_map: dict[int, int] = attrs.field(init=False, default=attrs.Factory(dict))
    source_filename: Optional[str] = None
    source_inds: Optional[np.ndarray] = None

    EXTS = ("h5", "hdf5", "slp")

    def __attrs_post_init__(self):
        # Check if the file accessible before applying heuristics.
        try:
            f = h5py.File(self.filename, "r")
        except OSError:
            return

        if self.dataset is None:
            # Iterate through datasets to find a rank 4 array.
            def find_movies(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                    self.dataset = name
                    return True

            f.visititems(find_movies)

            if self.dataset is None:
                return

        if isinstance(f[self.dataset], h5py.Group):
            if "video" in f[self.dataset]:
                self.dataset = f"{self.dataset}/video"

        if self.dataset.split("/")[-1] == "video":
            # This may be an embedded video dataset. Check for frame map.
            ds = f[self.dataset]
            if "frame_numbers" in ds.parent:
                frame_numbers = ds.parent["frame_numbers"][:]
                self.frame_map = {frame: idx for idx, frame in enumerate(frame_numbers)}
                self.source_inds = frame_numbers
                self.source_filename = json.loads(
                    ds.parent["source_video"].attrs["json"]
                )["backend"]["filename"]

        f.close()

    @property
    def num_frames(self) -> int:
        with h5py.File(self.filename, "r") as f:
            return f[self.dataset].shape[0]

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        with h5py.File(self.filename, "r") as f:
            ds = f[self.dataset]
            if "height" in ds.attrs:
                img_shape = (
                    ds.attrs["height"],
                    ds.attrs["width"],
                    ds.attrs["channels"],
                )
            else:
                img_shape = ds.shape[1:]
        if self.input_format == "channels_first":
            img_shape = img_shape[::-1]
        return img_shape

    def read_test_frame(self) -> np.ndarray:
        if self.frame_map:
            frame_idx = list(self.frame_map.keys())[0]
        else:
            frame_idx = 0
        return self.read_frame(frame_idx)

    def decode_embedded(self, img_string: np.ndarray, format: str) -> np.ndarray:
        if "cv2" in sys.modules:
            img = cv2.imdecode(img_string, cv2.IMREAD_UNCHANGED)
        else:
            img = iio.imread(BytesIO(img_string), extension=f".{format}")

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img

    def read_frame(self, frame_idx: int) -> np.ndarray:
        with h5py.File(self.filename, "r") as f:
            ds = f[self.dataset]

            if self.frame_map:
                frame_idx = self.frame_map[frame_idx]

            img = ds[frame_idx]

            if "format" in ds.attrs:
                img = self.decode_embedded(img, ds.attrs["format"])

        if self.input_format == "channels_first":
            img = np.transpose(img, (2, 1, 0))
        return img

    def read_frames(self, frame_inds: list) -> np.ndarray:
        with h5py.File(self.filename, "r") as f:
            if self.frame_map:
                frame_inds = [self.frame_map[idx] for idx in frame_inds]

            ds = f[self.dataset]
            imgs = ds[frame_inds]

            if "format" in ds.attrs:
                imgs = np.stack(
                    [self.decode_embedded(img, ds.attrs["format"]) for img in imgs],
                    axis=0,
                )

        if self.input_format == "channels_first":
            imgs = np.transpose(imgs, (2, 1, 0))

        return imgs
