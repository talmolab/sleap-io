"""Tests for Centroid data model."""

import numpy as np
import pytest

from sleap_io.model.bbox import PredictedBoundingBox, UserBoundingBox
from sleap_io.model.centroid import (
    Centroid,
    PredictedCentroid,
    UserCentroid,
    _geometric_median,
    get_centroid_skeleton,
)
from sleap_io.model.identity import Identity
from sleap_io.model.instance import Instance, PredictedInstance, Track
from sleap_io.model.mask import (
    PredictedSegmentationMask,
    UserSegmentationMask,
)
from sleap_io.model.roi import PredictedROI, UserROI
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
    assert c.track is None
    assert c.tracking_score is None
    assert c.instance is None
    assert c.category is None
    assert c.name == ""
    assert c.source == ""
    assert not c.is_predicted


def test_user_centroid_all_fields():
    track = Track(name="t1")
    c = UserCentroid(
        x=1.0,
        y=2.0,
        z=3.0,
        track=track,
        tracking_score=0.8,
        category="cell",
        name="c1",
        source="center_of_mass",
    )
    assert c.z == 3.0
    assert c.track is track
    assert c.tracking_score == 0.8
    assert c.category.name == "cell"
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


def test_centroid_is_empty():
    """is_empty reflects NaN coordinates."""
    assert not UserCentroid(x=1.0, y=2.0).is_empty
    assert UserCentroid(x=float("nan"), y=2.0).is_empty
    assert UserCentroid(x=1.0, y=float("nan")).is_empty
    assert UserCentroid(x=float("nan"), y=float("nan")).is_empty


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


def test_centroid_module_attr_unknown():
    """Accessing an unknown module attribute raises AttributeError."""
    from sleap_io.model import centroid

    with pytest.raises(AttributeError, match="no attribute"):
        _ = centroid.NOT_A_REAL_ATTR


# ---------------------------------------------------------------------------
# to_pose / from_pose (and deprecated to_instance / from_instance aliases)
# ---------------------------------------------------------------------------


def test_to_pose_user():
    c = UserCentroid(x=100.5, y=200.3, tracking_score=0.5)
    inst = c.to_pose()
    assert isinstance(inst, Instance)
    assert not isinstance(inst, PredictedInstance)
    pts = inst.numpy()
    assert pts.shape == (1, 2)
    np.testing.assert_allclose(pts[0], [100.5, 200.3])
    assert inst.tracking_score == 0.5


def test_to_pose_predicted():
    track = Track(name="t1")
    c = PredictedCentroid(x=50.0, y=60.0, score=0.9, track=track)
    inst = c.to_pose()
    assert isinstance(inst, PredictedInstance)
    pts = inst.numpy()
    np.testing.assert_allclose(pts[0], [50.0, 60.0])
    assert inst.score == 0.9
    assert inst.track is track


def test_to_pose_custom_skeleton():
    skel = Skeleton(["center"])
    c = UserCentroid(x=1.0, y=2.0)
    inst = c.to_pose(skeleton=skel)
    assert inst.skeleton is skel


def test_to_pose_multi_node_raises():
    skel = Skeleton(["a", "b"])
    c = UserCentroid(x=1.0, y=2.0)
    with pytest.raises(ValueError, match="exactly 1 node"):
        c.to_pose(skeleton=skel)


def test_to_instance_deprecated_alias():
    """to_instance is a deprecated alias forwarding to to_pose."""
    c = UserCentroid(x=3.0, y=4.0)
    with pytest.warns(DeprecationWarning, match="to_pose"):
        inst = c.to_instance()
    np.testing.assert_allclose(inst.numpy()[0], [3.0, 4.0])


def test_to_instance_deprecated_custom_skeleton():
    """Deprecated to_instance forwards the skeleton argument."""
    skel = Skeleton(["center"])
    c = UserCentroid(x=1.0, y=2.0)
    with pytest.warns(DeprecationWarning):
        inst = c.to_instance(skeleton=skel)
    assert inst.skeleton is skel


def test_from_pose_center_of_mass():
    skel = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skel,
        track=Track(name="t1"),
        tracking_score=0.7,
    )
    c = Centroid.from_pose(inst)
    assert isinstance(c, UserCentroid)
    assert c.x == pytest.approx(30.0)
    assert c.y == pytest.approx(40.0)
    assert c.track is inst.track
    assert c.tracking_score == 0.7
    assert c.instance is inst
    assert c.source == "center_of_mass"


def test_from_pose_center_of_mass_ignores_nan():
    """center_of_mass is a NaN-ignoring mean over visible nodes."""
    skel = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 20.0], [np.nan, np.nan]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="center_of_mass")
    assert c.x == pytest.approx(5.0)
    assert c.y == pytest.approx(10.0)


def test_from_pose_bbox_center():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [10.0, 20.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="bbox_center")
    assert c.x == pytest.approx(5.0)
    assert c.y == pytest.approx(10.0)
    assert c.source == "bbox_center"


def test_from_pose_geometric_median():
    """geometric_median of a symmetric square is the center."""
    skel = Skeleton(["a", "b", "c", "d"])
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="geometric_median")
    assert c.x == pytest.approx(1.0, abs=1e-4)
    assert c.y == pytest.approx(1.0, abs=1e-4)
    assert c.source == "geometric_median"


def test_from_pose_geometric_median_collinear():
    """geometric_median of collinear points is the middle point."""
    skel = Skeleton(["a", "b", "c"])
    inst = Instance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="geometric_median")
    assert c.x == pytest.approx(1.0, abs=1e-4)
    assert c.y == pytest.approx(0.0, abs=1e-4)


def test_geometric_median_known_value():
    """Direct geometric median of a known collinear triple."""
    pts = np.array([[0.0, 0.0], [5.0, 0.0], [100.0, 0.0]])
    x, y = _geometric_median(pts)
    assert x == pytest.approx(5.0, abs=1e-4)
    assert y == pytest.approx(0.0, abs=1e-4)


def test_geometric_median_coincident_points():
    """All-coincident points return that point (zero-distance branch)."""
    pts = np.array([[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]])
    x, y = _geometric_median(pts)
    assert x == pytest.approx(7.0)
    assert y == pytest.approx(9.0)


def test_geometric_median_iteration_cap():
    """Slow-converging input exits via the 100-iteration safety cap.

    This configuration does not reach the ``1e-6`` movement tolerance within
    100 Weiszfeld iterations, exercising the natural loop-exhaustion path. The
    result still approximates the true geometric median.
    """
    pts = np.array(
        [
            [0.74378048, -1.05813234],
            [4.92023229, 4.23745357],
            [-3.47992097, 0.89960593],
            [1.96215106, -3.63456586],
            [-1.87404353, 2.15917847],
            [4.01108093, -1.5825735],
        ]
    )
    x, y = _geometric_median(pts)
    assert x == pytest.approx(0.7663, abs=1e-2)
    assert y == pytest.approx(-1.0158, abs=1e-2)


def test_geometric_median_single_point():
    """A single point is its own geometric median."""
    pts = np.array([[3.0, 4.0]])
    x, y = _geometric_median(pts)
    assert x == pytest.approx(3.0)
    assert y == pytest.approx(4.0)


def test_from_pose_anchor():
    skel = Skeleton(["head", "thorax", "tail"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="anchor", node="thorax")
    assert c.x == 30.0
    assert c.y == 40.0
    assert c.source == "anchor:thorax"


def test_from_pose_anchor_by_index():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="anchor", node=1)
    assert c.x == 30.0
    assert c.y == 40.0
    assert c.source == "anchor:1"


def test_from_pose_anchor_by_numpy_index():
    """A numpy integer index is accepted for the anchor node."""
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="anchor", node=np.int64(0))
    assert c.x == 10.0
    assert c.y == 20.0


def test_from_pose_anchor_occluded_fallback():
    """Occluded anchor falls back to a non-anchor method with combined tag."""
    skel = Skeleton(["head", "thorax", "tail"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [np.nan, np.nan], [50.0, 60.0]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(
        inst, method="anchor", node="thorax", fallback="center_of_mass"
    )
    assert c.x == pytest.approx(30.0)
    assert c.y == pytest.approx(40.0)
    assert c.source == "anchor:thorax->center_of_mass"


def test_from_pose_anchor_occluded_no_fallback_degenerate():
    """Occluded anchor with no fallback yields a degenerate centroid."""
    skel = Skeleton(["head", "thorax"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [np.nan, np.nan]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst, method="anchor", node="thorax")
    assert c.is_empty
    assert c.source == "anchor:thorax"


def test_from_pose_anchor_occluded_no_fallback_error():
    """error_on_empty raises for an occluded anchor with no fallback."""
    skel = Skeleton(["head", "thorax"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [np.nan, np.nan]]),
        skeleton=skel,
    )
    with pytest.raises(ValueError, match="No visible points"):
        Centroid.from_pose(inst, method="anchor", node="thorax", error_on_empty=True)


def test_from_pose_anchor_fallback_exhausted():
    """Occluded anchor whose fallback also has no visible points is degenerate."""
    skel = Skeleton(["head", "thorax"])
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(
        inst, method="anchor", node="thorax", fallback="center_of_mass"
    )
    assert c.is_empty
    assert c.source == "anchor:thorax->center_of_mass"


def test_from_pose_anchor_fallback_exhausted_error():
    """Fallback-exhausted occluded anchor raises with error_on_empty."""
    skel = Skeleton(["head", "thorax"])
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        skeleton=skel,
    )
    with pytest.raises(ValueError, match="No visible points"):
        Centroid.from_pose(
            inst,
            method="anchor",
            node="thorax",
            fallback="center_of_mass",
            error_on_empty=True,
        )


def test_from_pose_anchor_invalid_fallback_raises():
    """An unknown fallback method raises when the anchor is occluded."""
    skel = Skeleton(["head", "thorax"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [np.nan, np.nan]]),
        skeleton=skel,
    )
    with pytest.raises(ValueError, match="Unknown method"):
        Centroid.from_pose(
            inst, method="anchor", node="thorax", fallback="not_a_method"
        )


def test_from_pose_anchor_no_node_raises():
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    with pytest.raises(ValueError, match="Must specify"):
        Centroid.from_pose(inst, method="anchor")


def test_from_pose_anchor_invalid_node_type_raises():
    """A non-str/int node specification raises."""
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    with pytest.raises(ValueError, match="node must be str or int"):
        Centroid.from_pose(inst, method="anchor", node=1.5)


def test_from_pose_predicted():
    skel = Skeleton(["a"])
    inst = PredictedInstance.from_numpy(
        np.array([[5.0, 10.0]]),
        skeleton=skel,
        score=0.85,
    )
    c = Centroid.from_pose(inst)
    assert isinstance(c, PredictedCentroid)
    assert c.score == 0.85


def test_from_pose_no_visible_degenerate():
    """No visible points yields a degenerate centroid by default."""
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        skeleton=skel,
    )
    c = Centroid.from_pose(inst)
    assert c.is_empty
    assert c.source == "center_of_mass"


def test_from_pose_no_visible_raises():
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        skeleton=skel,
    )
    with pytest.raises(ValueError, match="No visible points"):
        Centroid.from_pose(inst, error_on_empty=True)


def test_from_pose_bbox_center_no_visible_degenerate():
    """bbox_center with no visible points is degenerate."""
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[np.nan, np.nan]]), skeleton=skel)
    c = Centroid.from_pose(inst, method="bbox_center")
    assert c.is_empty


def test_from_pose_geometric_median_no_visible_degenerate():
    """geometric_median with no visible points is degenerate."""
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[np.nan, np.nan]]), skeleton=skel)
    c = Centroid.from_pose(inst, method="geometric_median")
    assert c.is_empty


def test_from_pose_invalid_method():
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    with pytest.raises(ValueError, match="Unknown method"):
        Centroid.from_pose(inst, method="invalid")


def test_from_pose_kwargs_passthrough():
    """Extra kwargs are forwarded to the centroid constructor."""
    skel = Skeleton(["a"])
    inst = Instance.from_numpy(np.array([[1.0, 2.0]]), skeleton=skel)
    c = Centroid.from_pose(inst, category="cell", name="c7")
    assert c.category.name == "cell"
    assert c.name == "c7"


def test_from_pose_carries_identity():
    """from_pose copies the instance's identity onto the centroid."""
    skel = Skeleton(["a", "b"])
    ident = Identity(name="cell_A")
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skel,
        identity=ident,
        identity_score=0.55,
    )
    c = UserCentroid.from_pose(inst)
    assert c.identity is ident
    assert c.identity_score == pytest.approx(0.55)


def test_from_instance_deprecated_alias():
    """from_instance is a deprecated alias forwarding to from_pose."""
    skel = Skeleton(["a", "b"])
    inst = Instance.from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        skel,
    )
    with pytest.warns(DeprecationWarning, match="from_pose"):
        c = Centroid.from_instance(inst)
    assert c.x == pytest.approx(20.0)
    assert c.y == pytest.approx(30.0)
    assert c.source == "center_of_mass"


def test_centroid_identity_field():
    """Centroid carries identity / identity_score (defaults None)."""
    ident = Identity(name="cell_A")
    c = UserCentroid(x=1.0, y=2.0, identity=ident, identity_score=0.7)
    assert c.identity is ident
    assert c.identity_score == pytest.approx(0.7)

    plain = UserCentroid(x=1.0, y=2.0)
    assert plain.identity is None
    assert plain.identity_score is None


def test_to_pose_carries_identity():
    """to_pose preserves identity / identity_score onto the instance."""
    ident = Identity(name="cell_A")
    c = UserCentroid(x=1.0, y=2.0, identity=ident, identity_score=0.6)
    inst = c.to_pose()
    assert inst.identity is ident
    assert inst.identity_score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# to_bbox
# ---------------------------------------------------------------------------


def test_to_bbox_scalar_size():
    c = UserCentroid(x=10.0, y=20.0, source="anchor:thorax")
    bb = c.to_bbox(size=4.0)
    assert isinstance(bb, UserBoundingBox)
    assert not isinstance(bb, PredictedBoundingBox)
    assert bb.x1 == pytest.approx(8.0)
    assert bb.y1 == pytest.approx(18.0)
    assert bb.x2 == pytest.approx(12.0)
    assert bb.y2 == pytest.approx(22.0)
    assert bb.angle == 0.0
    # Source metadata is inherited.
    assert bb.source == "anchor:thorax"


def test_to_bbox_tuple_size():
    c = UserCentroid(x=0.0, y=0.0)
    bb = c.to_bbox(size=(4.0, 6.0))
    assert bb.width == pytest.approx(4.0)
    assert bb.height == pytest.approx(6.0)


def test_to_bbox_padding():
    c = UserCentroid(x=0.0, y=0.0)
    bb = c.to_bbox(size=4.0, padding=1.0)
    assert bb.x1 == pytest.approx(-3.0)
    assert bb.x2 == pytest.approx(3.0)
    assert bb.y1 == pytest.approx(-3.0)
    assert bb.y2 == pytest.approx(3.0)


def test_to_bbox_size_required():
    c = UserCentroid(x=0.0, y=0.0)
    with pytest.raises(ValueError, match="'size' is required"):
        c.to_bbox(size=None)


def test_to_bbox_predicted():
    track = Track(name="t1")
    c = PredictedCentroid(x=5.0, y=5.0, score=0.7, track=track)
    bb = c.to_bbox(size=2.0)
    assert isinstance(bb, PredictedBoundingBox)
    assert bb.score == pytest.approx(0.7)
    assert bb.track is track


def test_to_bbox_empty_degenerate():
    c = UserCentroid(x=float("nan"), y=float("nan"))
    bb = c.to_bbox(size=4.0)
    assert bb.is_empty


def test_to_bbox_empty_error():
    c = UserCentroid(x=float("nan"), y=float("nan"))
    with pytest.raises(ValueError, match="degenerate"):
        c.to_bbox(size=4.0, error_on_empty=True)


# ---------------------------------------------------------------------------
# to_roi
# ---------------------------------------------------------------------------


def test_to_roi_basic():
    c = UserCentroid(x=10.0, y=20.0, category="cell")
    roi = c.to_roi(radius=3.0)
    assert isinstance(roi, UserROI)
    assert not roi.is_empty
    # Circle area ~= pi * r^2.
    assert roi.geometry.area == pytest.approx(np.pi * 9.0, abs=0.2)
    cx, cy = roi.geometry.centroid.x, roi.geometry.centroid.y
    assert cx == pytest.approx(10.0, abs=1e-6)
    assert cy == pytest.approx(20.0, abs=1e-6)
    assert roi.category.name == "cell"


def test_to_roi_predicted():
    c = PredictedCentroid(x=1.0, y=1.0, score=0.42)
    roi = c.to_roi(radius=2.0)
    assert isinstance(roi, PredictedROI)
    assert roi.score == pytest.approx(0.42)


def test_to_roi_empty_degenerate():
    c = UserCentroid(x=float("nan"), y=2.0)
    roi = c.to_roi(radius=3.0)
    assert roi.is_empty


def test_to_roi_empty_error():
    c = UserCentroid(x=float("nan"), y=2.0)
    with pytest.raises(ValueError, match="degenerate"):
        c.to_roi(radius=3.0, error_on_empty=True)


# ---------------------------------------------------------------------------
# to_mask
# ---------------------------------------------------------------------------


def test_to_mask_basic():
    c = UserCentroid(x=10.0, y=10.0)
    mask = c.to_mask(height=20, width=20, radius=3.0)
    assert isinstance(mask, UserSegmentationMask)
    assert not mask.is_empty
    # Rasterized disk area approximates pi * r^2.
    assert mask.area == pytest.approx(np.pi * 9.0, abs=6.0)


def test_to_mask_matches_roi_to_mask():
    """to_mask is equivalent to to_roi(radius).to_mask(h, w)."""
    c = UserCentroid(x=12.0, y=8.0)
    direct = c.to_mask(height=20, width=20, radius=4.0)
    via_roi = c.to_roi(radius=4.0).to_mask(20, 20)
    np.testing.assert_array_equal(direct.data, via_roi.data)


def test_to_mask_predicted():
    c = PredictedCentroid(x=5.0, y=5.0, score=0.33)
    mask = c.to_mask(height=12, width=12, radius=2.0)
    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.score == pytest.approx(0.33)


def test_to_mask_empty_degenerate():
    c = UserCentroid(x=float("nan"), y=float("nan"))
    mask = c.to_mask(height=8, width=8, radius=3.0)
    assert isinstance(mask, UserSegmentationMask)
    assert mask.is_empty
    assert mask.data.shape == (8, 8)


def test_to_mask_empty_predicted_degenerate():
    c = PredictedCentroid(x=float("nan"), y=float("nan"), score=0.9)
    mask = c.to_mask(height=8, width=8, radius=3.0)
    assert isinstance(mask, PredictedSegmentationMask)
    assert mask.is_empty
    assert mask.score == pytest.approx(0.9)


def test_to_mask_empty_error():
    c = UserCentroid(x=float("nan"), y=float("nan"))
    with pytest.raises(ValueError, match="degenerate"):
        c.to_mask(height=8, width=8, radius=3.0, error_on_empty=True)


def test_to_mask_metadata_propagated():
    """Metadata is carried through the to_mask conversion."""
    track = Track(name="t1")
    c = UserCentroid(x=10.0, y=10.0, track=track, category="cell", name="c1")
    mask = c.to_mask(height=20, width=20, radius=3.0)
    assert mask.track is track
    assert mask.category.name == "cell"
    assert mask.name == "c1"


# ---------------------------------------------------------------------------
# instance.to_centroid delegation
# ---------------------------------------------------------------------------


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
    assert c.instance is inst
