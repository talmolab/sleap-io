"""Media video backend reader for standard video formats (MP4, AVI, etc.)."""
from pims import PyAVReaderIndexed

# Ref: http://soft-matter.github.io/pims/v0.6.1/video.html
# vr = pims.Video(labels.videos[0].backend["filename"])
# vr.shape
# img = vr[0]
# img.shape


class MediaVideoReader(PyAVReaderIndexed):
    """Class for reading and manipulating frames of standard video formats (MP4, AVI, etc.).

    Attributes:
        file: The path of the video file as a string.
        height: The height of the video in pixels.
        width: The width of the video in pixels.
        channels: The number of color channels in the video.
        n_frames: The number of frames in the video.
    """

    def __init__(self, filename):
        super().__init__(file=filename)
        self.height = self.frame_shape[0]
        self.width = self.frame_shape[1]
        self.channels = self.frame_shape[2]
        self.n_frames = len(self)

def read_media_video(filename: str):
    return
