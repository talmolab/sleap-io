"""Data model for skeletons.

Skeletons are collections of nodes and edges which describe the landmarks associated
with a pose model. The edges represent the connections between them and may be used
differently depending on the underlying pose model.
"""

<<<<<<< HEAD
from __future__ import annotations
from attrs import define, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Text
import numpy as np
=======
from attrs import define
from typing import Optional, Tuple, List
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472


@define(auto_attribs=True)
class Node:
    """A landmark type within a `Skeleton`.

    This typically corresponds to a unique landmark within a skeleton, such as the "left
    eye".

    Attributes:
        name: Descriptive label for the landmark.
    """

    name: str

    @staticmethod
    def from_names(name_list: List[str]) -> List["Node"]:
        """Convert list of node names to list of nodes objects."""
        nodes = list()
        for name in name_list:
            nodes.append(Node(name))
        return nodes

<<<<<<< HEAD

=======
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
@define(auto_attribs=True)
class Edge:
    """A connection between two `Node` objects within a `Skeleton`.

    This is a directed edge, representing which `Node` comes first in the `Skeleton` tree.

    Attributes:
        source: The origin `Node`.
        destination: The destination `Node`.
    """

    source: Node
    destination: Node

    @staticmethod
    def from_names(edge_list: List[Tuple[str, str]]) -> List["Edge"]:
        edges = list()
        for edge in edge_list:
            edges.append(Edge(source=Node(edge[0]), destination=Node(edge[1])))
        return edges

<<<<<<< HEAD

=======
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
@define(auto_attribs=True)
class Skeleton:
    """A description of a set of landmark types and connections between them.

    Skeletons are represented by a directed graph composed of a set of `Node`s (landmark
    types such as body parts) and `Edge`s (connections between parts).

    Attributes:
        nodes: A list of `Node`s.
        edges: A list of `Edge`s.
        name: A descriptive name for the `Skeleton`.
        symmetries: A list of `Node` pairs corresponding to symmetries in the `Skeleton`.
    """

<<<<<<< HEAD
    nodes: list[Node]
    edges: list[Edge]
    name: Optional[str]
    symmetries: Optional[List[Tuple[Node, Node]]] = field(default=None)

    @staticmethod
    def from_names(nodes: List[str], edges: List[Tuple[str, str]]):

        return Skeleton(
            nodes=Node.from_names(nodes),
            edges=Edge.from_names(edges),
            name=None,
        )
=======
    nodes: List[Node]
    edges: List[Edge]
    name: Optional[str] = None
    symmetries: Optional[List[Tuple[Node, Node]]] = None
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
