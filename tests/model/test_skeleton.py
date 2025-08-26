"""Tests for methods in the sleap_io.model.skeleton file."""

import pytest

from sleap_io.model.skeleton import Edge, Node, Skeleton, Symmetry


def test_edge():
    """Test initialization and methods of `Edge` class."""
    edge = Edge(Node("A"), Node("B"))
    assert edge[0].name == "A"
    assert edge[1].name == "B"
    with pytest.raises(IndexError):
        edge[2]


def test_symmetry():
    """Test `Symmetry` class is initialized as expected."""
    A = Node("A")
    B = Node("B")

    s1 = Symmetry([A, B])
    s2 = Symmetry([B, A])
    assert s1 == s2


def test_skeleton():
    """Test initialization and methods of `Skeleton` object."""
    skel = Skeleton([Node("A"), Node("B")])
    assert skel.node_names == ["A", "B"]
    assert len(skel) == 2

    skel = Skeleton(["A", "B"])
    assert skel.node_names == ["A", "B"]
    for node in skel.nodes:
        assert type(node) is Node

    skel = Skeleton(["A", "B"], edges=[("A", "B")])
    assert skel.edges[0].source == skel.nodes[0]
    assert skel.edges[0].destination == skel.nodes[1]
    assert skel.edges[0] == Edge(skel.nodes[0], skel.nodes[1])
    assert skel.edge_inds == [(0, 1)]

    assert str(skel) == 'Skeleton(nodes=["A", "B"], edges=[(0, 1)])'

    with pytest.raises(IndexError):
        skel[None]

    with pytest.raises(IndexError):
        skel.index(None)

    with pytest.raises(ValueError):
        Skeleton(["A", "B"], edges=[("a", "B")])

    with pytest.raises(ValueError):
        Skeleton(["A", "B"], edges=[("A", "C")])

    skel = Skeleton(["A", "B"], symmetries=[("A", "B")])
    assert skel.symmetry_inds == [(0, 1)]

    with pytest.raises(ValueError):
        Skeleton(["A", "B"], symmetries=[("a", "B")])

    with pytest.raises(ValueError):
        Skeleton(["A", "B"], symmetries=[("A", "b")])


def test_skeleton_node_map():
    """Test `Skeleton` node map returns correct nodes."""
    A = Node("A")
    B = Node("B")
    skel = Skeleton([A, B])

    assert skel.index("A") == 0
    assert skel.index("B") == 1

    assert skel.index(A) == 0
    assert skel.index(B) == 1

    assert skel["A"] == A
    assert skel["B"] == B

    skel.nodes = [B, A]
    assert skel.index("A") == 1
    assert skel.index("B") == 0


def test_get_flipped_node_inds():
    skel = Skeleton(["A", "BL", "BR", "C", "DL", "DR"])
    assert skel.get_flipped_node_inds() == [0, 1, 2, 3, 4, 5]

    skel.add_symmetry("BL", "BR")
    skel.add_symmetry("DL", "DR")
    assert skel.get_flipped_node_inds() == [0, 2, 1, 3, 5, 4]

    assert skel.symmetries[0][0].name in ("BL", "BR")
    assert skel.symmetries[0][1].name in ("BL", "BR")
    syms = list(skel.symmetries[0])
    assert syms[0] != syms[1]


def test_edge_unpack():
    skel = Skeleton(["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    edge = skel.edges[0]
    assert edge[0].name == "A"
    assert edge[1].name == "B"

    src, dst = skel.edges[0]
    assert src.name == "A"
    assert dst.name == "B"


def test_add_node():
    skel = Skeleton()
    skel.add_node("A")
    assert skel.node_names == ["A"]
    assert "A" in skel
    assert skel.index("A") == 0

    B = Node("B")
    skel.add_node(B)
    assert skel.node_names == ["A", "B"]
    assert B in skel
    assert "B" in skel
    assert skel.index("B") == 1

    skel.add_node("C")
    assert skel.node_names == ["A", "B", "C"]

    with pytest.raises(ValueError):
        skel.add_node("B")

    skel.add_nodes(["D", "E"])
    assert skel.node_names == ["A", "B", "C", "D", "E"]


def test_add_edge():
    skel = Skeleton(["A", "B"])
    skel.add_edge("A", "B")
    assert skel.edge_inds == [(0, 1)]
    assert skel.edge_names == [("A", "B")]

    skel.add_edge("B", "A")
    assert skel.edge_inds == [(0, 1), (1, 0)]

    skel.add_edge("A", "B")
    assert skel.edge_inds == [(0, 1), (1, 0)]

    skel.add_edge("A", "C")
    assert skel.edge_inds == [(0, 1), (1, 0), (0, 2)]

    skel.add_edge("D", "A")
    assert skel.edge_inds == [(0, 1), (1, 0), (0, 2), (3, 0)]

    skel.add_edges([("D", "E"), ("E", "A")])
    assert skel.edge_inds == [(0, 1), (1, 0), (0, 2), (3, 0), (3, 4), (4, 0)]


def test_add_symmetry():
    skel = Skeleton(["A", "B"])
    skel.add_symmetry("A", "B")
    assert skel.symmetry_inds == [(0, 1)]
    assert skel.symmetry_names == [("A", "B")]

    # Don't duplicate reversed symmetries
    skel.add_symmetry("B", "A")
    assert skel.symmetry_inds == [(0, 1)]
    assert skel.symmetry_names == [("A", "B")]

    # Add new symmetry with new node objects
    skel.add_symmetry(Symmetry([Node("C"), Node("D")]))
    assert skel.symmetry_inds == [(0, 1), (2, 3)]

    # Add new symmetry with node names
    skel.add_symmetry("E", "F")
    assert skel.symmetry_inds == [(0, 1), (2, 3), (4, 5)]

    # Add symmetries
    skel.add_nodes(["GL", "GR", "HL", "HR"])
    skel.add_symmetries([("GL", "GR"), ("HL", "HR")])
    assert skel.symmetry_inds == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def test_no_duplicate_symmetries():
    """Test that duplicate symmetry edges are not created."""
    skel = Skeleton(["A", "B", "C", "D", "E", "F"])

    # Add initial symmetry
    skel.add_symmetry("A", "B")
    assert len(skel.symmetries) == 1
    assert skel.symmetry_inds == [(0, 1)]

    # Try adding the same symmetry again (should not duplicate)
    skel.add_symmetry("A", "B")
    assert len(skel.symmetries) == 1
    assert skel.symmetry_inds == [(0, 1)]

    # Try adding the reversed symmetry (should not duplicate)
    skel.add_symmetry("B", "A")
    assert len(skel.symmetries) == 1
    assert skel.symmetry_inds == [(0, 1)]

    # Add a different symmetry
    skel.add_symmetry("C", "D")
    assert len(skel.symmetries) == 2
    assert skel.symmetry_inds == [(0, 1), (2, 3)]

    # Try adding duplicates using add_symmetries (batch operation)
    skel.add_symmetries([("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")])
    # Should still only have 2 symmetries
    assert len(skel.symmetries) == 2
    assert skel.symmetry_inds == [(0, 1), (2, 3)]

    # Add new symmetry and verify count
    skel.add_symmetry("E", "F")
    assert len(skel.symmetries) == 3
    assert skel.symmetry_inds == [(0, 1), (2, 3), (4, 5)]

    # Try adding a mix of new and duplicate symmetries
    initial_count = len(skel.symmetries)
    skel.add_symmetries([("A", "B"), ("E", "F")])  # Both duplicates
    assert len(skel.symmetries) == initial_count  # No change

    # Verify using Node objects directly (not just strings)
    node_a = skel["A"]
    node_b = skel["B"]
    skel.add_symmetry(node_a, node_b)
    assert len(skel.symmetries) == 3  # Still no duplicate

    # Verify symmetry set behavior
    # Note: Symmetry uses sets internally, order doesn't matter for nodes
    sym1 = Symmetry([Node("X"), Node("Y")])
    # The nodes are stored as a set, so we can verify the set contains both nodes
    assert len(sym1.nodes) == 2
    node_names = {node.name for node in sym1.nodes}
    assert node_names == {"X", "Y"}


def test_rename_nodes():
    """Test renaming nodes in the skeleton."""
    skel = Skeleton(["A", "B", "C"])
    skel.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
    assert skel.node_names == ["X", "Y", "Z"]

    skel.rename_nodes(["a", "b", "c"])
    assert skel.node_names == ["a", "b", "c"]

    skel.rename_node("a", "A")
    assert skel.node_names == ["A", "b", "c"]

    # Incorrect length when passing a list
    with pytest.raises(ValueError):
        skel.rename_nodes(["a1", "b1"])

    # Target node already exists
    with pytest.raises(ValueError):
        skel.rename_nodes({"b": "c"})

    # Source node doesn't exist
    with pytest.raises(ValueError):
        skel.rename_nodes({"d": "e"})


def test_remove_nodes():
    skel = Skeleton(["A", "B", "C", "D", "EL", "ER"])
    skel.add_edges([("A", "B"), ("B", "C"), ("B", "D")])
    skel.add_symmetry("EL", "ER")
    assert skel.edge_inds == [(0, 1), (1, 2), (1, 3)]
    assert skel.symmetry_inds == [(4, 5)]

    skel.remove_nodes(["A", "C"])
    assert skel.node_names == ["B", "D", "EL", "ER"]
    assert skel.index("B") == 0
    assert skel.index("D") == 1
    assert skel.index("EL") == 2
    assert skel.index("ER") == 3
    assert "A" not in skel
    assert "C" not in skel
    assert skel.edge_inds == [(0, 1)]
    assert skel.edge_names == [("B", "D")]
    assert skel.symmetry_inds == [(2, 3)]

    skel.remove_node("B")
    assert skel.node_names == ["D", "EL", "ER"]
    assert skel.edge_inds == []
    assert skel.symmetry_inds == [(1, 2)]

    skel.remove_node("ER")
    assert skel.node_names == ["D", "EL"]
    assert skel.symmetry_inds == []

    with pytest.raises(IndexError):
        skel.remove_nodes(["ER"])


def test_reorder_nodes():
    skel = Skeleton(["A", "B", "C"])
    skel.add_edges([("A", "B"), ("B", "C")])
    assert skel.edge_inds == [(0, 1), (1, 2)]

    skel.reorder_nodes(["C", "A", "B"])
    assert skel.node_names == ["C", "A", "B"]
    assert skel.index("C") == 0
    assert skel.index("A") == 1
    assert skel.index("B") == 2
    assert skel.edge_names == [("A", "B"), ("B", "C")]
    assert skel.edge_inds == [(1, 2), (2, 0)]

    # Incorrect length
    with pytest.raises(ValueError):
        skel.reorder_nodes(["C", "A"])

    # Node not in skeleton
    with pytest.raises(IndexError):
        skel.reorder_nodes(["C", "A", "X"])


def test_skeleton_matches():
    """Test Skeleton.matches() method."""
    # Create test skeletons
    skel1 = Skeleton(
        nodes=["head", "thorax", "abdomen", "left_wing", "right_wing"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
        symmetries=[("left_wing", "right_wing")],
    )

    skel2 = Skeleton(
        nodes=["head", "thorax", "abdomen", "left_wing", "right_wing"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
        symmetries=[("left_wing", "right_wing")],
    )

    skel3 = Skeleton(
        nodes=[
            "abdomen",
            "thorax",
            "head",
            "left_wing",
            "right_wing",
        ],  # Different order
        edges=[("head", "thorax"), ("thorax", "abdomen")],
        symmetries=[("left_wing", "right_wing")],
    )

    skel4 = Skeleton(
        nodes=["head", "thorax", "tail"],  # Different node
        edges=[("head", "thorax"), ("thorax", "tail")],
    )

    skel5 = Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "abdomen"), ("thorax", "abdomen")],  # Different edges
    )

    # Test exact match (same order required)
    assert skel1.matches(skel2, require_same_order=True)
    assert not skel1.matches(skel3, require_same_order=True)

    # Test structure match (order doesn't matter)
    assert skel1.matches(skel2, require_same_order=False)
    assert skel1.matches(skel3, require_same_order=False)

    # Test with different nodes
    assert not skel1.matches(skel4, require_same_order=False)

    # Test with different edges
    assert not skel1.matches(skel5, require_same_order=False)

    # Test with different number of nodes
    skel6 = Skeleton(nodes=["head", "thorax"])
    assert not skel1.matches(skel6, require_same_order=False)


def test_skeleton_node_similarities():
    """Test Skeleton.node_similarities() method."""
    skel1 = Skeleton(nodes=["head", "thorax", "abdomen"])
    skel2 = Skeleton(nodes=["head", "thorax", "tail"])
    skel3 = Skeleton(nodes=["wing1", "wing2", "antenna"])
    skel4 = Skeleton(nodes=["head", "thorax", "abdomen"])  # Same as skel1

    # Test partial overlap
    metrics = skel1.node_similarities(skel2)
    assert metrics["n_common"] == 2  # head and thorax
    assert metrics["n_self_only"] == 1  # abdomen
    assert metrics["n_other_only"] == 1  # tail
    assert metrics["jaccard"] == 2 / 4  # 2 common / 4 total unique
    assert metrics["dice"] == 2 * 2 / (3 + 3)  # 2*2 / (3+3)

    # Test no overlap
    metrics = skel1.node_similarities(skel3)
    assert metrics["n_common"] == 0
    assert metrics["n_self_only"] == 3
    assert metrics["n_other_only"] == 3
    assert metrics["jaccard"] == 0
    assert metrics["dice"] == 0

    # Test complete overlap
    metrics = skel1.node_similarities(skel4)
    assert metrics["n_common"] == 3
    assert metrics["n_self_only"] == 0
    assert metrics["n_other_only"] == 0
    assert metrics["jaccard"] == 1.0
    assert metrics["dice"] == 1.0

    # Test with empty skeleton
    skel_empty = Skeleton(nodes=[])
    metrics = skel1.node_similarities(skel_empty)
    assert metrics["n_common"] == 0
    assert metrics["n_self_only"] == 3
    assert metrics["n_other_only"] == 0
    assert metrics["jaccard"] == 0
    assert metrics["dice"] == 0


def test_skeleton_matches_edge_mismatch():
    """Test Skeleton.matches() with edge mismatches."""
    # Test edge mismatch case (line 720 in skeleton.py)
    skel1 = Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )

    # Same nodes, different edges
    skel2 = Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "abdomen")],  # Different edge structure
    )

    # Should not match in STRUCTURE mode due to different edges
    assert not skel1.matches(skel2, require_same_order=False)

    # Test with more edges in other skeleton
    skel3 = Skeleton(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen"), ("head", "abdomen")],
    )

    # Different number of edges
    assert not skel1.matches(skel3, require_same_order=False)


def test_skeleton_matches_edge_set_difference():
    """Test that edge set comparison triggers return False at line 731."""
    # Create skeleton with specific edges
    skel1 = Skeleton(
        nodes=["A", "B", "C", "D"], edges=[("A", "B"), ("B", "C"), ("C", "D")]
    )

    # Same nodes but completely different edge connections
    skel2 = Skeleton(
        nodes=["A", "B", "C", "D"],
        edges=[("A", "C"), ("B", "D"), ("A", "D")],  # Same count, different connections
    )

    # This should trigger line 731: if self_edge_set != other_edge_set: return False
    assert not skel1.matches(skel2, require_same_order=False)

    # Try with one edge swapped
    skel3 = Skeleton(
        nodes=["A", "B", "C", "D"],
        edges=[("A", "B"), ("C", "B"), ("C", "D")],  # B->C changed to C->B
    )

    assert not skel1.matches(skel3, require_same_order=False)

    # Try with edges in reverse direction
    skel4 = Skeleton(
        nodes=["A", "B", "C", "D"],
        edges=[("B", "A"), ("C", "B"), ("D", "C")],  # All edges reversed
    )

    assert not skel1.matches(skel4, require_same_order=False)


def test_skeleton_matches_exact_edge_case():
    """Test edge case: same node/edge count but different edges."""
    # Tests line 731 in skeleton.py: if self_edge_set != other_edge_set

    # Create a linear skeleton
    linear = Skeleton(
        nodes=["n1", "n2", "n3", "n4"],
        edges=[("n1", "n2"), ("n2", "n3"), ("n3", "n4")],  # Linear chain
    )

    # Create a branched skeleton with same number of edges
    branched = Skeleton(
        nodes=["n1", "n2", "n3", "n4"],
        edges=[("n1", "n2"), ("n1", "n3"), ("n1", "n4")],  # Star pattern from n1
    )

    # Both have 4 nodes and 3 edges, but different connectivity
    assert len(linear.nodes) == len(branched.nodes)
    assert len(linear.edges) == len(branched.edges)

    # Should not match due to different edge sets (line 731)
    assert not linear.matches(branched, require_same_order=False)

    # Also test with a ring vs linear
    ring = Skeleton(
        nodes=["n1", "n2", "n3", "n4"],
        edges=[
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n1"),
        ],  # Forms a triangle with n1,n2,n3
    )

    assert not linear.matches(ring, require_same_order=False)


def test_skeleton_matches_comprehensive_edge_coverage():
    """Comprehensive test to ensure all edge comparison paths are covered."""
    # Base skeleton
    base = Skeleton(nodes=["a", "b", "c"], edges=[("a", "b"), ("b", "c")])

    # Test 1: Same edges, different node order (match with require_same_order=False)
    reordered = Skeleton(
        nodes=["b", "c", "a"],  # Different order
        edges=[("a", "b"), ("b", "c")],  # Same edges
    )
    assert base.matches(reordered, require_same_order=False)
    assert not base.matches(reordered, require_same_order=True)

    # Test 2: Different edges with same nodes (triggers line 731)
    diff_edges = Skeleton(
        nodes=["a", "b", "c"],
        edges=[("a", "c"), ("b", "c")],  # Different edge pattern
    )
    assert not base.matches(diff_edges, require_same_order=False)

    # Test 3: Subset of edges (different count, caught earlier)
    subset = Skeleton(
        nodes=["a", "b", "c"],
        edges=[("a", "b")],  # Only one edge
    )
    assert not base.matches(subset, require_same_order=False)

    # Test 4: Superset of edges (different count)
    superset = Skeleton(
        nodes=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c"), ("a", "c")],  # Extra edge
    )
    assert not base.matches(superset, require_same_order=False)

    # Test 5: Empty edges
    no_edges1 = Skeleton(nodes=["a", "b", "c"], edges=[])
    no_edges2 = Skeleton(nodes=["a", "b", "c"], edges=[])
    assert no_edges1.matches(no_edges2, require_same_order=False)


def test_skeleton_matches_edge_comparison_coverage():
    """Ensure edge comparison specifically triggers line 731."""
    # Create skeletons that will have same nodes, same edge count,
    # but different edge sets to trigger line 731

    skel_triangle = Skeleton(
        nodes=["A", "B", "C"],
        edges=[("A", "B"), ("B", "C"), ("C", "A")],  # Forms a triangle
    )

    skel_chain = Skeleton(
        nodes=["A", "B", "C"],
        edges=[("A", "B"), ("B", "C"), ("A", "C")],  # Different connections
    )

    # Both have 3 nodes and 3 edges
    assert len(skel_triangle.nodes) == len(skel_chain.nodes) == 3
    assert len(skel_triangle.edges) == len(skel_chain.edges) == 3

    # But the edge sets are different, should trigger line 731
    result = skel_triangle.matches(skel_chain, require_same_order=False)
    assert result is False  # Explicitly check the return value

    # Also verify the edge sets are indeed different
    tri_edges = {(e.source.name, e.destination.name) for e in skel_triangle.edges}
    chain_edges = {(e.source.name, e.destination.name) for e in skel_chain.edges}
    assert tri_edges != chain_edges  # Confirm sets are different


def test_skeleton_matches_symmetry_mismatch():
    """Test Skeleton.matches() with symmetry mismatches."""
    # Test symmetry mismatch cases (lines 731, 735 in skeleton.py)

    # Test case 1: Different number of symmetries (line 735)
    skel1 = Skeleton(
        nodes=["head", "thorax", "left_wing", "right_wing"],
        edges=[("head", "thorax")],
        symmetries=[("left_wing", "right_wing")],
    )

    skel2 = Skeleton(
        nodes=["head", "thorax", "left_wing", "right_wing"],
        edges=[("head", "thorax")],
        symmetries=[],  # No symmetries
    )

    # Should not match due to different number of symmetries
    assert not skel1.matches(skel2, require_same_order=False)

    # Test case 2: Same number of symmetries but different content
    skel3 = Skeleton(
        nodes=["head", "thorax", "left_wing", "right_wing", "left_leg", "right_leg"],
        edges=[("head", "thorax")],
        symmetries=[("left_wing", "right_wing"), ("left_leg", "right_leg")],
    )

    skel4 = Skeleton(
        nodes=["head", "thorax", "left_wing", "right_wing", "left_leg", "right_leg"],
        edges=[("head", "thorax")],
        symmetries=[("left_wing", "right_wing")],  # Only one symmetry
    )

    # Should not match due to different number of symmetries
    assert not skel3.matches(skel4, require_same_order=False)

    # Same number but different symmetry groups
    skel3 = Skeleton(
        nodes=["left_eye", "right_eye", "nose"],
        edges=[],
        symmetries=[["left_eye", "nose"]],  # Different pairing
    )

    assert not skel1.matches(skel3, require_same_order=False)
