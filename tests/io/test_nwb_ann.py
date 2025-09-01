"""Tests for NWB annotations functionality."""

from __future__ import annotations

import numpy as np
from ndx_pose import Skeleton as NdxSkeleton

from sleap_io import Skeleton
from sleap_io.io.nwb_ann import (
    nwb_skeleton_to_sleap_skeleton,
    sleap_skeleton_to_nwb_skeleton,
)


def test_sleap_skeleton_to_nwb_skeleton_basic():
    """Test basic conversion from sleap-io Skeleton to ndx-pose Skeleton."""
    # Create sleap-io skeleton
    sleap_skeleton = Skeleton(
        nodes=["nose", "head", "neck", "body"],
        edges=[("nose", "head"), ("head", "neck"), ("neck", "body")],
        name="test_skeleton",
    )

    # Test conversion
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    # Verify nodes
    assert list(nwb_skeleton.nodes) == ["nose", "head", "neck", "body"]

    # Verify edges (should be array of indices)
    expected_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(nwb_skeleton.edges, expected_edges)

    # Verify name
    assert nwb_skeleton.name == "test_skeleton"


def test_sleap_skeleton_to_nwb_skeleton_no_name():
    """Test conversion when sleap skeleton has no name."""
    sleap_skeleton = Skeleton(nodes=["a", "b"], edges=[("a", "b")])

    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    # Should use default name
    assert nwb_skeleton.name == "skeleton"


def test_sleap_skeleton_to_nwb_skeleton_empty():
    """Test conversion with empty skeleton."""
    sleap_skeleton = Skeleton(nodes=[], edges=[], name="empty")

    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(sleap_skeleton)

    assert list(nwb_skeleton.nodes) == []
    assert nwb_skeleton.edges.shape == (0, 2)
    assert nwb_skeleton.name == "empty"


def test_nwb_skeleton_to_sleap_skeleton_basic():
    """Test basic conversion from ndx-pose Skeleton to sleap-io Skeleton."""
    # Create ndx-pose skeleton
    nwb_skeleton = NdxSkeleton(
        name="test_skeleton",
        nodes=["nose", "head", "neck", "body"],
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint8),
    )

    # Convert to sleap-io
    sleap_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Verify nodes
    assert sleap_skeleton.node_names == ["nose", "head", "neck", "body"]

    # Verify edges
    expected_edges = [(0, 1), (1, 2), (2, 3)]
    assert sleap_skeleton.edge_inds == expected_edges

    # Verify name
    assert sleap_skeleton.name == "test_skeleton"


def test_nwb_skeleton_to_sleap_skeleton_empty():
    """Test conversion with empty ndx-pose skeleton."""
    nwb_skeleton = NdxSkeleton(
        name="empty", nodes=[], edges=np.array([], dtype=np.uint8).reshape(0, 2)
    )

    sleap_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    assert sleap_skeleton.node_names == []
    assert sleap_skeleton.edge_inds == []
    assert sleap_skeleton.name == "empty"


def test_skeleton_roundtrip_conversion():
    """Test that roundtrip conversion preserves skeleton structure."""
    # Create original sleap-io skeleton
    original_skeleton = Skeleton(
        nodes=["nose", "head", "neck", "left_shoulder", "right_shoulder", "body"],
        edges=[
            ("nose", "head"),
            ("head", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("neck", "body"),
        ],
        name="complex_skeleton",
    )

    # Convert to ndx-pose and back
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(original_skeleton)
    recovered_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Verify structure is preserved
    assert recovered_skeleton.node_names == original_skeleton.node_names
    assert recovered_skeleton.edge_inds == original_skeleton.edge_inds
    assert recovered_skeleton.name == original_skeleton.name

    # Verify skeletons match
    assert recovered_skeleton.matches(original_skeleton, require_same_order=True)


def test_skeleton_roundtrip_with_symmetries():
    """Test that skeleton conversion handles symmetries correctly."""
    # Create skeleton with symmetries
    original_skeleton = Skeleton(
        nodes=["nose", "left_eye", "right_eye", "body"],
        edges=[("nose", "left_eye"), ("nose", "right_eye"), ("nose", "body")],
        symmetries=[("left_eye", "right_eye")],
        name="symmetric_skeleton",
    )

    # Convert to ndx-pose and back
    nwb_skeleton = sleap_skeleton_to_nwb_skeleton(original_skeleton)
    recovered_skeleton = nwb_skeleton_to_sleap_skeleton(nwb_skeleton)

    # Note: ndx-pose doesn't support symmetries, so they will be lost
    assert recovered_skeleton.node_names == original_skeleton.node_names
    assert recovered_skeleton.edge_inds == original_skeleton.edge_inds
    assert recovered_skeleton.name == original_skeleton.name
    assert len(recovered_skeleton.symmetries) == 0  # Symmetries are lost
