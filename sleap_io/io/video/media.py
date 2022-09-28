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
        video_shape: The shape of the video as a tuple (height, width, channels, frames)

    Examples:
        >>> video = Video('video.avi')  # or .mov, etc.
        >>> imshow(video[0]) # Show the first frame.
        >>> imshow(video[-1]) # Show the last frame.
        >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

        >>> for frame in video[:]:
        ...    # Do something with every frame.

        >>> for frame in video[10:20]:
        ...    # Do something with frames 10-20.

        >>> for frame in video[[5, 7, 13]]:
        ...    # Do something with frames 5, 7, and 13.

        >>> video_shape = video.video_shape  # Dimensions of video (height, width, channels, frames)
        >>> frame_shape = video.frame_shape  # Pixel dimensions of video (height, width, channels)
    """

    def __init__(self, filename):
        super().__init__(file=filename)
        self.video_shape = self.frame_shape + (len(self),)

    @classmethod
    def read_media_video(cls, filename: str):
        return cls(filename)
