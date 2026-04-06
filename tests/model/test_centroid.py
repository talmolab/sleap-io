"""Tests for Centroid data model."""

import numpy as np
import pytest

from sleap_io.model.centroid import (
    Centroid,
    PredictedCentroid,
    UserCentroid,
    get_centroid_skeleton,
)
from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.skeleton import Skeleton


def test_centroid_abstract():
    """Centroid cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Centroid is abstract"):
        Centroid(x=1.0, y=2.0)


def test_user_centroid_basic():
    c = UserCentroid(x=10.5, y=20.3)
    assert c.x == 10.5
    assert c.y == 20.3
    assert c.z is None
    assert c.video is None
    assert c.frame_idx is None
    assert c.track is None
    assert c.tracking_score is None
    assert c.instance is None
    assert c.category == ""
    assert c.name == ""
    assert c.source == ""
    assert not c.is_predicted


def test_user_centroid_all_fields():
    track = Track(name="t1")
    c = UserCentroid(
        x=1.0,
        y=2.0,
        z=3.0,
        frame_idx=5,
        track=track,
        tracking_score=0.8,
        category="cell",
        name="c1",
        source="center_of_mass",
    )
    assert c.z == 3.0
    assert c.frame_idx == 5
    assert c.track is track
    assert c.tracking_score == 0.8
    assert c.category == "cell"
    assert c.name == "c1"
    assert c.source == "center_of_mass"


def test_predicted_centroid():
    c = PredictedCentroid(x=1.0, y=2.0, score=0.95)
    assert c.score == 0.95
    assert c.is_predicted


def test_predicted_centroid_default_score():
    c = PredictedCentroid(x=1.0, y=2.0)
    assert c.score == 0.0


def test_centroid_identity_equality():
    c1 = UserCentroid(x=1.0, y=2.0)
    c2 = UserCentroid(x=1.0, y=2.0)
    assert c1 is not c2
    assert c1 != c2


def test_centroid_xy():
    c = UserCentroid(x=10.0, y=20.0)
    assert c.xy == (10.0, 20.0)


def test_centroid_yx():
    c = UserCentroid(x=10.0, y=20.0)
    assert c.yx == (20.0, 10.0)


def test_centroid_xyz():
    c = UserCentroid(x=1.0, y=2.0, z=3.0)
    assert c.xyz == (1.0, 2.0, 3.0)

    c2 = UserCentroid(x=1.0, y=2.0)
    assert c2.xyz == (1.0, 2.0, None)


def test_centroid_skeleton():
    skel = get_centroid_skeleton()
    assert len(skel) == 1
    assert skel.node_names == ["centroid"]
    # Same object on repeated calls.
    assert get_centroid_skeleton() is skel


def test_centroid_module_attr():
    from sleap_io.model import centroid

    skel = centroid.CENTROID_SKELETON
    assert len(skel) == 1
    assert skel.node_names == ["centroid"]


def test_to_instance_user():
    c = UserCentroid(x=100.5, y=200.3, tracking_score=0.5)
    inst = c.to_instance()
    assert isinstance(inst, Instance)
    assert not isinstance(inst, PredictedInstance)
    pts = inst.numpy()
    assert pts.shape == (1, 2)
    np.testing.assert_allclose(pts[0], [100.5, 200.3])
    assert inst.tracking_score == 0.5


def test_to_instance_predicted():
    track = Track(name="t1")
    c = PredictedCentroid(x=50.0, y=60.0, score=0.9, track=track)
    inst = c.to_instance()
    assert isinstance(inst, PredictedInstance)
    pts = inst.numpy()
    np.testing.assert_allclose(pts[0], [50.0, 60.0])
    assert inst.score == 0.9
    assert inst.track is track


def test_to_instance_custom_skeleton():
    skel = Skeleton(["center"])
    c = UserCentroid(x=1.0, y=2.0)
    inst = c.to_instance(skeleton=skel)
    assert inst.skeleton is skel


def test_to_instance_multi_node_raises():
    skel = Skeleton(["a", "b"])
    c = UserCentroid(x=1.0, y=2.0)
    with pytest.raises(ValueError, match="exactly 1 node"):
        c.to_instance(skeleton=skel)


def test_from_instance_center_of_mass():
    skel = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skel,
        track=Track(name="t1"),
        tracking_score=0.7,
    )
    c = Centroid.from_instance(inst)
    assert isinstance(c, UserCentroid)
    assert c.x == pytest.approx(30.0)
    assert c.y == pytest.approx(40.0)
    assert c.track is inst.track
    assert c.tracking_score == 0.7
    assert c.instance is inst
    assert c.source == "center_of_mass"


def test_from_instance_bbox_center():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 20.0]]),
        skeleton=skel,
    )
    c = Centroid.from_instance(inst, method="bbox_center")
    assert c.x == pytest.approx(5.0)
    assert c.y == pytest.approx(10.0)
    assert c.source == "bbox_center"


def test_from_instance_anchor():
    skel = Skeleton(["head", "thorax", "tail"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skel,
    )
    c = Centroid.from_instance(inst, method="anchor", node="thorax")
    assert c.x == 30.0
    assert c.y == 40.0
    assert c.source == "anchor:thorax"


def test_from_instance_anchor_by_index():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skel,
    )
    c = Centroid.from_instance(inst, method="anchor", node=1)
    assert c.x == 30.0
    assert c.y == 40.0


def test_from_instance_predicted():
    skel = Skeleton(["a"])
    inst = PredictedInstance.from_numpy(
        np.array([[5.0, 10.0]]),
        skeleton=skel,
        score=0.85,
    )
    c = Centroid.from_instance(inst)
    assert isinstance(c, PredictedCentroid)
    assert c.score == 0.85


def test_from_instance_no_visible_raises():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        skeleton=skel,
    )
    with pytest.raises(ValueError, match="No visible points"):
        Centroid.from_instance(inst)


def test_from_instance_invalid_method():
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    with pytest.raises(ValueError, match="Unknown method"):
        Centroid.from_instance(inst, method="invalid")


def test_from_instance_anchor_no_node_raises():
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    with pytest.raises(ValueError, match="Must specify"):
        Centroid.from_instance(inst, method="anchor")


def test_instance_centroid_xy():
    skel = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0], [np.nan, np.nan]]),
        skeleton=skel,
    )
    cx, cy = inst.centroid_xy
    assert cx == pytest.approx(20.0)
    assert cy == pytest.approx(30.0)


def test_instance_centroid_xy_no_visible():
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[np.nan, np.nan]]), skeleton=skel)
    assert inst.centroid_xy is None


def test_instance_to_centroid():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skel,
    )
    c = inst.to_centroid()
    assert isinstance(c, UserCentroid)
    assert c.x == pytest.approx(20.0)
    assert c.y == pytest.approx(30.0)
