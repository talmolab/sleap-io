from numpy.testing import assert_equal
import pytest
from sleap_io import Video, Skeleton, Instance, PredictedInstance, LabeledFrame
from sleap_io.model.labels import Labels


def test_labels():
    labels = Labels(
        [
            LabeledFrame(
                video=Video(filename="test", shape=(1, 1, 1, 1)),
                frame_idx=0,
                instances=[
                    Instance([[0, 1], [2, 3]], skeleton=Skeleton(["A", "B"])),
                    PredictedInstance([[4, 5], [6, 7]], skeleton=Skeleton(["A", "B"])),
                ],
            )
        ]
    )

    assert len(labels) == 1
    assert type(labels[0]) == LabeledFrame
    assert labels[0].frame_idx == 0

    with pytest.raises(IndexError):
        labels[None]
