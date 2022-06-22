"""Data model for skeletons.

Skeletons are collections of nodes and edges which describe the landmarks associated
with a pose model. The edges represent the connections between them and may be used
differently depending on the underlying pose model.
"""

from __future__ import annotations
from attrs import define, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Text
from itertools import count

# import networkx as nx
import numpy as np

NodeRef = Union[str, "Node"]
skeleton_idx = count(0)


@define
class Node:
    """A landmark type within a `Skeleton`.

    This typically corresponds to a unique landmark within a skeleton, such as the "left
    eye".

    Attributes:
        name: Descriptive label for the landmark.
    """

    name: str = field()

    @staticmethod
    def from_names(name_list: List[str]) -> List["Node"]:
        """Convert list of node names to list of nodes objects."""
        nodes = []
        for name in name_list:
            nodes.append(Node(name))
        return nodes


@define
class Edge:
    """A connection between two nodes within a `Skeleton`.

    This is a directed edge, representing which node comes first in the skeleton tree.

    Attributes:
        source: The origin `Node`.
        destination: The destination `Node`.
    """

    source: Node = field()
    destination: Node = field()

    @staticmethod
    def from_names(edge_list: List[Tuple[str]]) -> List["Edge"]:
        edges = list()
        for edge in edge_list:
            edges.append(Edge(source=Node(edge[0]), destination=Node(edge[1])))
        return edges


@define
class Symmetry:
    """SYMMETRY - these edges represent symmetrical relationships
    between parts (e.g. left and right arms)
    """

    pair: Tuple[Node, Node] = field()


@define
class Skeleton:
    """A description of a set of landmark types and connections between them.

    Skeletons are represented by a directed graph composed of a set of `Node`s (landmark
    types such as body parts) and `Edge`s (connections between parts).

    Attributes:
        nodes: A list of `Node`s.
        edges: A list of `Edge`s.
        name: A descriptive name for the skeleton.
    """

    _skeleton_idx = count(0)

    nodes: list[Node] = field()
    edges: list[Edge] = field()
    symmetries: Optional[list[Symmetry]] = field(default=None)
    name: Optional[str] = field(default=None)

    @staticmethod
    def from_names(nodes: List[str], edges: List[Tuple[str]]):

        return Skeleton(
            nodes=Node.from_names(nodes),
            edges=Edge.from_names(edges),
            name="Skeleton-" + str(next(skeleton_idx)),
        )

    # def graph(self):
    #     graph = nx.MultiDiGraph()
    #     for Node in self.nodes:
    #         graph.add_node(Node.name)
    #     for Edge in self.edges:
    #         graph.add_edge(Edge.source.name, Edge.destination.name)
    #     graph.name = self.name
    #     return graph


# e.g.

# skeleton = Skeleton.from_names(
#     nodes=["head", "thorax", "abdomen"],
#     edges=[("head", "thorax"), ("thorax", "abdomen")]
# )

# print(skeleton)
