"""Data model for skeletons.

Skeletons are collections of nodes and edges which describe the landmarks associated
with a pose model. The edges represent the connections between them and may be used
differently depending on the underlying pose model.
"""

from __future__ import annotations
from attrs import define, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Text
import numpy as np


@define(auto_attribs=True)
class Node:
    """A landmark type within a `Skeleton`.

    This typically corresponds to a unique landmark within a skeleton, such as the "left
    eye".

    Attributes:
        name: Descriptive label for the landmark.
    """

    name: str


@define(auto_attribs=True)
class Edge:
    """A connection between two `Node` objects within a `Skeleton`.

    This is a directed edge, representing which node comes first in the skeleton tree.

    Attributes:
        source: The origin `Node`.
        destination: The destination `Node`.
    """

    source: Node
    destination: Node


@define(auto_attribs=True)
class Skeleton:
    """A description of a set of landmark types and connections between them.

    Skeletons are represented by a directed graph composed of a set of `Node`s (landmark
    types such as body parts) and `Edge`s (connections between parts).

    Attributes:
        nodes: A list of `Node`s.
        edges: A list of `Edge`s.
        name: A descriptive name for the skeleton.
    """

    nodes: list[Node]
    edges: list[Edge]
    name: Optional[str]
    symmetries: Optional[List[Tuple[Node, Node]]] = field(default=None)
