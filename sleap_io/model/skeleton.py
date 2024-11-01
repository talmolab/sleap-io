"""Data model for skeletons.

Skeletons are collections of nodes and edges which describe the landmarks associated
with a pose model. The edges represent the connections between them and may be used
differently depending on the underlying pose model.
"""

from __future__ import annotations
from attrs import define, field
from typing import TypeAlias
import numpy as np


@define(eq=False)
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

    def __getitem__(self, idx) -> Node:
        """Return the source `Node` (`idx` is 0) or destination `Node` (`idx` is 1)."""
        if idx == 0:
            return self.source
        elif idx == 1:
            return self.destination
        else:
            raise IndexError("Edge only has 2 nodes (source and destination).")


@define
class Symmetry:
    """A relationship between a pair of nodes denoting their left/right pairing.

    Attributes:
        nodes: A set of two `Node`s.
    """

    nodes: set[Node] = field(converter=set, validator=lambda _, __, val: len(val) == 2)

    def __iter__(self):
        """Iterate over the symmetric nodes."""
        return iter(self.nodes)

    def __getitem__(self, idx) -> Node:
        """Return the first node."""
        for i, node in enumerate(self.nodes):
            if i == idx:
                return node


NodeOrIndex: TypeAlias = Node | str | int


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

    nodes: list[Node] = field(
        factory=list,
        on_setattr=lambda self, attr, new_nodes: self.rebuild_cache(nodes=new_nodes),
    )
    edges: list[Edge] = field(factory=list)
    symmetries: list[Symmetry] = field(factory=list)
    name: str | None = None
    _name_to_node_cache: dict[str, Node] = field(init=False, repr=False, eq=False)
    _node_to_ind_cache: dict[Node, int] = field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self):
        """Ensure nodes are `Node`s, edges are `Edge`s, and `Node` map is updated."""
        self._convert_nodes()
        self._convert_edges()
        self.rebuild_cache()

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

    def rebuild_cache(self, nodes: list[Node] | None = None):
        """Rebuild the node name/index to `Node` map caches.

        Args:
            nodes: A list of `Node` objects to update the cache with. If not provided,
                the cache will be updated with the current nodes in the skeleton. If
                nodes are provided, the cache will be updated with the provided nodes,
                but the current nodes in the skeleton will not be updated. Default is
                `None`.

        Notes:
            This function should be called when nodes or node list is mutated to update
            the lookup caches for indexing nodes by name or `Node` object.

            This is done automatically when nodes are added or removed from the skeleton
            using the convenience methods in this class.

            This method only needs to be used when manually mutating nodes or the node
            list directly.
        """
        if nodes is None:
            nodes = self.nodes
        self._name_to_node_cache = {node.name: node for node in nodes}
        self._node_to_ind_cache = {node: i for i, node in enumerate(nodes)}

    @property
    def node_names(self) -> list[str]:
        """Names of the nodes associated with this skeleton as a list of strings."""
        return [node.name for node in self.nodes]

    @property
    def edge_inds(self) -> list[tuple[int, int]]:
        """Edges indices as a list of 2-tuples."""
        return [
            (self.nodes.index(edge.source), self.nodes.index(edge.destination))
            for edge in self.edges
        ]

    @property
    def edge_names(self) -> list[str, str]:
        """Edge names as a list of 2-tuples with string node names."""
        return [(edge.source.name, edge.destination.name) for edge in self.edges]

    @property
    def symmetry_inds(self) -> list[tuple[int, int]]:
        """Symmetry indices as a list of 2-tuples."""
        return [
            tuple(sorted((self.index(symmetry[0]), self.index(symmetry[1]))))
            for symmetry in self.symmetries
        ]

    @property
    def symmetry_names(self) -> list[str, str]:
        """Symmetry names as a list of 2-tuples with string node names."""
        return [
            (self.nodes[i].name, self.nodes[j].name) for (i, j) in self.symmetry_inds
        ]

    def get_flipped_node_inds(self) -> list[int]:
        """Returns node indices that should be switched when horizontally flipping.

        This is useful as a lookup table for flipping the landmark coordinates when
        doing data augmentation.

        Example:
            >>> skel = Skeleton(["A", "B_left", "B_right", "C", "D_left", "D_right"])
            >>> skel.add_symmetry("B_left", "B_right")
            >>> skel.add_symmetry("D_left", "D_right")
            >>> skel.flipped_node_inds
            [0, 2, 1, 3, 5, 4]
            >>> pose = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
            >>> pose[skel.flipped_node_inds]
            array([[0, 0],
                   [2, 2],
                   [1, 1],
                   [3, 3],
                   [5, 5],
                   [4, 4]])
        """
        flip_idx = np.arange(len(self.nodes))
        if len(self.symmetries) > 0:
            symmetry_inds = np.array(
                [(self.index(a), self.index(b)) for a, b in self.symmetries]
            )
            flip_idx[symmetry_inds[:, 0]] = symmetry_inds[:, 1]
            flip_idx[symmetry_inds[:, 1]] = symmetry_inds[:, 0]

        flip_idx = flip_idx.tolist()
        return flip_idx

    def __len__(self) -> int:
        """Return the number of nodes in the skeleton."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """Return a readable representation of the skeleton."""
        nodes = ", ".join([f'"{node}"' for node in self.node_names])
        return "Skeleton(" f"nodes=[{nodes}], " f"edges={self.edge_inds}" ")"

    def index(self, node: Node | str) -> int:
        """Return the index of a node specified as a `Node` or string name."""
        if type(node) == str:
            return self.index(self._name_to_node_cache[node])
        elif type(node) == Node:
            return self._node_to_ind_cache[node]
        else:
            raise IndexError(f"Invalid indexing argument for skeleton: {node}")

    def __getitem__(self, idx: NodeOrIndex) -> Node:
        """Return a `Node` when indexing by name or integer."""
        if type(idx) == int:
            return self.nodes[idx]
        elif type(idx) == str:
            return self._name_to_node_cache[idx]
        else:
            raise IndexError(f"Invalid indexing argument for skeleton: {idx}")

    def __contains__(self, node: NodeOrIndex) -> bool:
        """Check if a node is in the skeleton."""
        if type(node) == str:
            return node in self._name_to_node_cache
        elif type(node) == Node:
            return node in self.nodes
        elif type(node) == int:
            return 0 <= node < len(self.nodes)
        else:
            raise ValueError(f"Invalid node type for skeleton: {node}")

    def add_node(self, node: Node | str):
        """Add a `Node` to the skeleton.

        Args:
            node: A `Node` object or a string name to create a new node.
        """
        if node in self:
            raise ValueError(f"Node '{node}' already exists in the skeleton.")

        if type(node) == str:
            node = Node(node)

        self.nodes.append(node)

        # Atomic update of the cache.
        self._name_to_node_cache[node.name] = node
        self._node_to_ind_cache[node] = len(self.nodes) - 1

    def add_nodes(self, nodes: list[Node | str]):
        """Add multiple `Node`s to the skeleton.

        Args:
            nodes: A list of `Node` objects or string names to create new nodes.
        """
        for node in nodes:
            self.add_node(node)

    def require_node(self, node: NodeOrIndex) -> Node:
        """Return a `Node` if it exists in the skeleton, otherwise add it.

        Args:
            node: A `Node` object, name or index.

        Returns:
            The `Node` object.
        """
        if node not in self:
            self.add_node(node)
        if type(node) == Node:
            return node
        return self[node]

    def add_edge(
        self,
        src: NodeOrIndex | Edge | tuple[NodeOrIndex, NodeOrIndex],
        dst: NodeOrIndex | None = None,
    ):
        """Add an `Edge` to the skeleton.

        Args:
            src: The source node specified as a `Node`, name or index.
            dst: The destination node specified as a `Node`, name or index.
        """
        edge = None
        if type(src) == tuple:
            src, dst = src

        if isinstance(src, NodeOrIndex):
            if not isinstance(dst, NodeOrIndex):
                raise ValueError("Destination node must be specified.")

            src = self.require_node(src)
            dst = self.require_node(dst)
            edge = Edge(src, dst)

        if type(src) == Edge:
            edge = src

        if edge not in self.edges:
            self.edges.append(edge)

    def add_edges(self, edges: list[Edge | tuple[NodeOrIndex, NodeOrIndex]]):
        """Add multiple `Edge`s to the skeleton.

        Args:
            edges: A list of `Edge` objects or 2-tuples of source and destination nodes.
        """
        for edge in edges:
            self.add_edge(edge)

    def add_symmetry(
        self, node1: Symmetry | NodeOrIndex = None, node2: NodeOrIndex | None = None
    ):
        """Add a symmetry relationship to the skeleton.

        Args:
            node1: The first node specified as a `Node`, name or index. If a `Symmetry`
                object is provided, it will be added directly to the skeleton.
            node2: The second node specified as a `Node`, name or index.
        """
        symmetry = None
        if type(node1) == Symmetry:
            symmetry = node1
            node1, node2 = symmetry

        node1 = self.require_node(node1)
        node2 = self.require_node(node2)

        if symmetry is None:
            symmetry = Symmetry({node1, node2})

        if symmetry not in self.symmetries:
            self.symmetries.append(symmetry)

    def rename_nodes(self, name_map: dict[NodeOrIndex, str] | list[str]):
        """Rename nodes in the skeleton.

        Args:
            name_map: A dictionary mapping old node names to new node names. Keys can be
                specified as `Node` objects, integer indices, or string names. Values
                must be specified as string names.

                If a list of strings is provided of the same length as the current
                nodes, the nodes will be renamed to the names in the list in order.

        Notes:
            This method should always be used when renaming nodes in the skeleton as it
            handles updating the lookup caches necessary for indexing nodes by name.

            After renaming, instances using this skeleton do NOT need to be updated as
            the nodes are stored by reference in the skeleton.

        Example:
            >>> skel = Skeleton(["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
            >>> skel.rename_nodes({"A": "X", "B": "Y", "C": "Z"})
            >>> skel.node_names
            ["X", "Y", "Z"]
            >>> skel.rename_nodes(["a", "b", "c"])
            >>> skel.node_names
            ["a", "b", "c"]

        Raises:
            ValueError: If the new node names exist in the skeleton or if the old node
                names are not found in the skeleton.
        """
        if type(name_map) == list:
            if len(name_map) != len(self.nodes):
                raise ValueError(
                    "List of new node names must be the same length as the current "
                    "nodes."
                )
            name_map = {node: name for node, name in zip(self.nodes, name_map)}

        for old_name, new_name in name_map.items():
            if type(old_name) == Node:
                old_name = old_name.name
            if type(old_name) == int:
                old_name = self.nodes[old_name].name

            if old_name not in self._name_to_node_cache:
                raise ValueError(f"Node '{old_name}' not found in the skeleton.")
            if new_name in self._name_to_node_cache:
                raise ValueError(f"Node '{new_name}' already exists in the skeleton.")

            node = self._name_to_node_cache[old_name]
            node.name = new_name
            self._name_to_node_cache[new_name] = node
            del self._name_to_node_cache[old_name]

    def rename_node(self, old_name: NodeOrIndex, new_name: str):
        """Rename a single node in the skeleton.

        Args:
            old_name: The name of the node to rename. Can also be specified as an
                integer index or `Node` object.
            new_name: The new name for the node.
        """
        self.rename_nodes({old_name: new_name})

    def remove_nodes(self, nodes: list[NodeOrIndex]):
        """Remove nodes from the skeleton.

        Args:
            nodes: A list of node names, indices, or `Node` objects to remove.

        Notes:
            This method should always be used when removing nodes from the skeleton as
            it handles updating the lookup caches necessary for indexing nodes by name.

            It also removes any edges and symmetries that are connected to the removed
            nodes.
        """
        # Standardize input and make a pre-mutation copy before keys are changed.
        rm_node_inds = [
            self.index(node) if type(node) != int else node for node in nodes
        ]
        rm_node_objs = [self.nodes[ind] for ind in rm_node_inds]

        # Remove nodes from the skeleton.
        for node in rm_node_objs:
            self.nodes.remove(node)
            del self._name_to_node_cache[node.name]

        # Update node index map.
        self.rebuild_cache()

        # Remove edges connected to the removed nodes.
        self.edges = [
            edge
            for edge in self.edges
            if edge.source not in rm_node_objs and edge.destination not in rm_node_objs
        ]

        # Remove symmetries connected to the removed nodes.
        self.symmetries = [
            symmetry
            for symmetry in self.symmetries
            if symmetry.nodes.isdisjoint(rm_node_objs)
        ]
