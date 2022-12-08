"""Data model for skeletons.

Skeletons are collections of nodes and edges which describe the landmarks associated
with a pose model. The edges represent the connections between them and may be used
differently depending on the underlying pose model.
"""

from __future__ import annotations
from attrs import define, field
from typing import Optional, Tuple, Union
import numpy as np


@define(frozen=True, cache_hash=True)
class Node:
    """A landmark type within a `Skeleton`.

    This typically corresponds to a unique landmark within a skeleton, such as the "left
    eye".

    Attributes:
        name: Descriptive label for the landmark.
    """

    name: str


@define(frozen=True)
class Edge:
    """A connection between two `Node` objects within a `Skeleton`.

    This is a directed edge, representing the ordering of `Node`s in the `Skeleton`
    tree.

    Attributes:
        source: The origin `Node`.
        destination: The destination `Node`.
    """

    source: Node
    destination: Node

    def __getitem__(self, idx):
        """Return the source `Node` (`idx` is 0) or destination `Node` (`idx` is 1)."""
        if idx == 0:
            return self.source
        elif idx == 1:
            return self.destination
        else:
            raise ValueError("Edge only has 2 nodes (source and destination).")


@define
class Symmetry:
    """A relationship between a pair of nodes denoting their left/right pairing.

    Attributes:
        nodes: A set of two `Node`s.
    """

    nodes: set[Node] = field(converter=set, validator=lambda _, __, val: len(val) == 2)


@define
class Skeleton:
    """A description of a set of landmark types and connections between them.

    Skeletons are represented by a directed graph composed of a set of `Node`s (landmark
    types such as body parts) and `Edge`s (connections between parts).

    Attributes:
        nodes: A list of `Node`s. May be specified as a list of strings to create new
            nodes from their names.
        edges: A list of `Edge`s. May be specified as a list of 2-tuples of string names
            or integer indices of `nodes`. Each edge corresponds to a pair of source and
            destination nodes forming a directed edge.
        symmetries: A list of `Symmetry`s. Each symmetry corresponds to symmetric body
            parts, such as `"left eye", "right eye"`. This is used when applying flip
            (reflection) augmentation to images in order to appropriately swap the
            indices of symmetric landmarks.
        name: A descriptive name for the `Skeleton`.
    """

    def _update_node_map(self, attr, nodes):
        """Callback for maintaining node name/index to `Node` map."""
        self._node_name_map = {node.name: node for node in nodes}
        self._node_ind_map = {node: i for i, node in enumerate(nodes)}

    nodes: list[Node] = field(on_setattr=_update_node_map)
    edges: list[Edge] = field(factory=list)
    symmetries: list[Symmetry] = field(factory=list)
    name: Optional[str] = None
    _node_name_map: dict[str, Node] = field(init=False, repr=False, eq=False)
    _node_ind_map: dict[Node, int] = field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self):
        """Ensure nodes are `Node`s, edges are `Edge`s, and `Node` map is updated."""
        self._convert_nodes()
        self._convert_edges()
        self._update_node_map(None, self.nodes)

    def _convert_nodes(self):
        """Convert nodes to `Node` objects if needed."""
        if isinstance(self.nodes, np.ndarray):
            object.__setattr__(self, "nodes", self.nodes.tolist())
        for i, node in enumerate(self.nodes):
            if type(node) == str:
                self.nodes[i] = Node(node)

    def _convert_edges(self):
        """Convert list of edge names or integers to `Edge` objects if needed."""
        if isinstance(self.edges, np.ndarray):
            self.edges = self.edges.tolist()
        node_names = self.node_names
        for i, edge in enumerate(self.edges):
            if type(edge) == Edge:
                continue
            src, dst = edge
            if type(src) == str:
                try:
                    src = node_names.index(src)
                except ValueError:
                    raise ValueError(
                        f"Node '{src}' specified in the edge list is not in the nodes."
                    )
            if type(src) == int or (
                np.isscalar(src) and np.issubdtype(src.dtype, np.integer)
            ):
                src = self.nodes[src]

            if type(dst) == str:
                try:
                    dst = node_names.index(dst)
                except ValueError:
                    raise ValueError(
                        f"Node '{dst}' specified in the edge list is not in the nodes."
                    )
            if type(dst) == int or (
                np.isscalar(dst) and np.issubdtype(dst.dtype, np.integer)
            ):
                dst = self.nodes[dst]

            self.edges[i] = Edge(src, dst)

    @property
    def node_names(self) -> list[str]:
        """Names of the nodes associated with this skeleton as a list of strings."""
        return [node.name for node in self.nodes]

    @property
    def edge_inds(self) -> list[Tuple[int, int]]:
        """Edges indices as a list of 2-tuples."""
        return [
            (self.nodes.index(edge.source), self.nodes.index(edge.destination))
            for edge in self.edges
        ]

    def __len__(self) -> int:
        """Return the number of nodes in the skeleton."""
        return len(self.nodes)

    def index(self, node: Union[Node, str]) -> int:
        """Return the index of a node specified as a `Node` or string name."""
        if type(node) == str:
            return self.index(self._node_name_map[node])
        elif type(node) == Node:
            return self._node_ind_map[node]
        else:
            raise IndexError(f"Invalid indexing argument for skeleton: {node}")

    def __getitem__(self, idx: Union[int, str]) -> Node:
        """Return a `Node` when indexing by name or integer."""
        if type(idx) == int:
            return self.nodes[idx]
        elif type(idx) == str:
            return self._node_name_map[idx]
        else:
            raise IndexError(f"Invalid indexing argument for skeleton: {idx}")
