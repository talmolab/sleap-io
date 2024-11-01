"""Tests for methods in the sleap_io.model.skeleton file."""

import pytest
from sleap_io.model.skeleton import Node, Edge, Symmetry, Skeleton


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
        assert type(node) == Node

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

    skel.add_node(Node("B"))
    assert skel.node_names == ["A", "B"]

    skel.add_node("C")
    assert skel.node_names == ["A", "B", "C"]

    with pytest.raises(ValueError):
        skel.add_node("B")
        assert skel.node_names == ["A", "B", "C"]


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


def test_rename_nodes():
    """Test renaming nodes in the skeleton."""
    skel = Skeleton(["A", "B", "C"])
    skel.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
    assert skel.node_names == ["X", "Y", "Z"]

    skel.rename_nodes(["a", "b", "c"])
    assert skel.node_names == ["a", "b", "c"]

    with pytest.raises(ValueError):
        skel.rename_nodes(["a1", "b1"])

    with pytest.raises(ValueError):
        skel.rename_nodes({"a": "b"})

    with pytest.raises(ValueError):
        skel.rename_nodes({"d": "e"})


def test_rename_node():
    """Test renaming a single node in the skeleton."""
    skel = Skeleton(["A", "B", "C"])
    skel.rename_node("A", "X")
    assert skel.node_names == ["X", "B", "C"]

    skel.rename_node(1, "Y")
    assert skel.node_names == ["X", "Y", "C"]

    skel.rename_node(skel.nodes[2], "Z")
    assert skel.node_names == ["X", "Y", "Z"]

    with pytest.raises(ValueError):
        skel.rename_node("X", "Y")

    with pytest.raises(ValueError):
        skel.rename_node("D", "E")
