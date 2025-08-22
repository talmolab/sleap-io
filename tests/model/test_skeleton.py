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
    # Note: Symmetry uses sets internally, so order doesn't matter for the nodes collection
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
