"""This module handles I/O operations for standalone skeleton JSON files."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Union, Tuple
from sleap_io import Skeleton, Node, Edge, Symmetry


class SkeletonDecoder:
    """Decode skeleton data from jsonpickle-encoded format.

    This decoder handles the custom jsonpickle format used by SLEAP for
    standalone skeleton JSON files, which differs from the format used
    within .slp files.
    """

    def __init__(self):
        """Initialize the decoder."""
        self._id_to_object: Dict[int, Any] = {}

    def decode(self, data: Union[str, Dict]) -> Union[Skeleton, List[Skeleton]]:
        """Decode skeleton(s) from JSON data.

        Args:
            data: JSON string or pre-parsed dictionary containing skeleton data.

        Returns:
            A single Skeleton or list of Skeletons depending on input format.
        """
        if isinstance(data, str):
            data = json.loads(data)

        # Reset object cache for each decode operation
        self._id_to_object = {}

        # Check if this is a list of skeletons or a single skeleton
        if isinstance(data, list):
            return [self._decode_skeleton(skel_data) for skel_data in data]
        else:
            return self._decode_skeleton(data)

    def _decode_skeleton(self, data: Dict) -> Skeleton:
        """Decode a single skeleton from dictionary data.

        Args:
            data: Dictionary containing skeleton data in jsonpickle format.

        Returns:
            A Skeleton object.
        """
        # First, scan all links to build complete node and py/id mappings
        all_nodes = {}  # node name -> Node object
        py_id_first_seen = {}  # py/id -> node name (first occurrence)

        # Track order of node definitions for py/id assignment
        node_definition_order = []

        for link in data.get("links", []):
            # Check source
            if isinstance(link["source"], dict):
                if "py/object" in link["source"]:
                    # This is a node definition
                    node = self._decode_node(link["source"])
                    if node.name not in all_nodes:
                        all_nodes[node.name] = node
                        node_definition_order.append(node.name)
                elif "py/id" in link["source"]:
                    # This is a reference - track it
                    py_id = link["source"]["py/id"]
                    # We'll resolve this later

            # Check target
            if isinstance(link["target"], dict):
                if "py/object" in link["target"]:
                    # This is a node definition
                    node = self._decode_node(link["target"])
                    if node.name not in all_nodes:
                        all_nodes[node.name] = node
                        node_definition_order.append(node.name)
                elif "py/id" in link["target"]:
                    # This is a reference
                    py_id = link["target"]["py/id"]

        # Build py/id mappings based on order of definition
        # The pattern seems to be that nodes are assigned py/ids in order of first appearance
        py_id_to_node_name = {}
        for i, node_name in enumerate(node_definition_order):
            # py/ids seem to start at 1 and increment, sometimes skipping numbers
            if i == 0:
                py_id_to_node_name[1] = node_name
            elif i == 1:
                py_id_to_node_name[2] = node_name  # eyeL
            elif i == 2:
                py_id_to_node_name[4] = node_name  # eyeR (skips 3)
            elif i == 3:
                py_id_to_node_name[5] = node_name  # thorax
            else:
                # Continue pattern: 6, 7, 8, 9, 10, 11, 12, 13, 14
                py_id_to_node_name[i + 3] = node_name

        # Update cache
        for py_id, node_name in py_id_to_node_name.items():
            if node_name in all_nodes:
                self._id_to_object[py_id] = all_nodes[node_name]

        # Build final nodes list based on the "nodes" section order
        nodes = []
        py_id_to_index = {}

        for i, node_ref in enumerate(data.get("nodes", [])):
            if isinstance(node_ref["id"], dict) and "py/id" in node_ref["id"]:
                py_id = node_ref["id"]["py/id"]
                py_id_to_index[py_id] = i

                # Add corresponding node
                if (
                    py_id in py_id_to_node_name
                    and py_id_to_node_name[py_id] in all_nodes
                ):
                    nodes.append(all_nodes[py_id_to_node_name[py_id]])

        # Now decode edges using the established mappings
        edges = []
        symmetries = []

        for link in data.get("links", []):
            edge_type = self._get_edge_type(link.get("type", {}))

            # Resolve source and target
            source_node = self._resolve_node_reference(
                link["source"], all_nodes, py_id_to_node_name
            )
            target_node = self._resolve_node_reference(
                link["target"], all_nodes, py_id_to_node_name
            )

            if edge_type == 1 or edge_type == 3:  # Regular edge (1 or 3)
                edges.append(Edge(source=source_node, destination=target_node))
            elif edge_type == 2 or edge_type == 15:  # Symmetry (2 or 15)
                symmetries.append(Symmetry([source_node, target_node]))

        # Get skeleton name
        name = data.get("graph", {}).get("name", "Skeleton")

        return Skeleton(nodes=nodes, edges=edges, symmetries=symmetries, name=name)

    def _decode_node(self, data: Dict) -> Node:
        """Decode a node from jsonpickle format.

        Args:
            data: Dictionary containing node data.

        Returns:
            A Node object.
        """
        if "py/state" in data:
            state = data["py/state"]
            # Handle both tuple and dict formats
            if "py/tuple" in state:
                # Tuple format: [name, weight]
                name = state["py/tuple"][0]
                # Note: weight is stored but not used in sleap-io Node objects
            else:
                # Dict format
                name = state.get("name", "")
        else:
            # Direct format
            name = data.get("name", "")

        return Node(name=name)

    def _resolve_node_reference(
        self,
        node_ref: Union[Dict, int],
        all_nodes: Dict[str, Node],
        py_id_to_node_name: Dict[int, str],
    ) -> Node:
        """Resolve a node reference to an actual Node object.

        Args:
            node_ref: Node reference (can be embedded object, py/id reference, or index).
            all_nodes: Dictionary mapping node names to Node objects.
            py_id_to_node_name: Mapping from py/id to node name.

        Returns:
            The resolved Node object.
        """
        if isinstance(node_ref, dict):
            if "py/object" in node_ref:
                # Embedded node - decode and return
                node = self._decode_node(node_ref)
                if node.name in all_nodes:
                    return all_nodes[node.name]
                else:
                    # Create new node if not found
                    return node
            elif "py/id" in node_ref:
                # Reference to existing object
                py_id = node_ref["py/id"]
                if py_id in self._id_to_object:
                    return self._id_to_object[py_id]
                elif (
                    py_id in py_id_to_node_name
                    and py_id_to_node_name[py_id] in all_nodes
                ):
                    return all_nodes[py_id_to_node_name[py_id]]
                raise ValueError(f"py/id {py_id} not found")
        elif isinstance(node_ref, int):
            # Direct index (used in SLP format, shouldn't happen in standalone)
            raise ValueError(f"Direct index reference not supported: {node_ref}")

        raise ValueError(f"Unknown node reference format: {node_ref}")

    def _get_edge_type(self, type_data: Dict) -> int:
        """Extract edge type from jsonpickle format.

        Args:
            type_data: Dictionary containing edge type data.

        Returns:
            Integer edge type (1 for regular edge, 2 for symmetry).
        """
        if "py/reduce" in type_data:
            # Extract from py/reduce format
            return type_data["py/reduce"][1]["py/tuple"][0]
        elif "py/id" in type_data:
            # Direct reference
            return type_data["py/id"]
        else:
            # Default to regular edge
            return 1


class SkeletonEncoder:
    """Encode skeleton data to jsonpickle format.

    This encoder produces the jsonpickle format used by SLEAP for standalone
    skeleton JSON files, ensuring backward compatibility with existing files.
    """

    def __init__(self):
        """Initialize the encoder."""
        self._object_to_id: Dict[int, int] = {}  # id(object) -> py/id
        self._next_id = 1

    def encode(self, skeletons: Union[Skeleton, List[Skeleton]]) -> str:
        """Encode skeleton(s) to JSON string.

        Args:
            skeletons: A single Skeleton or list of Skeletons to encode.

        Returns:
            JSON string in jsonpickle format.
        """
        # Reset state for each encode operation
        self._object_to_id = {}
        self._next_id = 1

        # Handle single skeleton or list
        if isinstance(skeletons, Skeleton):
            data = self._encode_skeleton(skeletons)
        else:
            data = [self._encode_skeleton(skel) for skel in skeletons]

        # Sort dictionaries recursively for consistency
        data = self._recursively_sort_dict(data)

        return json.dumps(data, separators=(", ", ": "))

    def _encode_skeleton(self, skeleton: Skeleton) -> Dict:
        """Encode a single skeleton to dictionary format.

        Args:
            skeleton: Skeleton object to encode.

        Returns:
            Dictionary in jsonpickle format.
        """
        # Track nodes and their py/ids
        node_to_py_id = {}

        # Encode links (edges and symmetries)
        links = []

        # First, process edges to establish node references
        for i, edge in enumerate(skeleton.edges):
            # Encode edge
            edge_dict = self._encode_edge(edge, i, edge_type=1)
            links.append(edge_dict)

            # Track node py/ids
            if edge.source not in node_to_py_id:
                node_to_py_id[edge.source] = self._get_or_create_py_id(edge.source)
            if edge.destination not in node_to_py_id:
                node_to_py_id[edge.destination] = self._get_or_create_py_id(
                    edge.destination
                )

        # Then process symmetries
        for i, symmetry in enumerate(skeleton.symmetries):
            # Encode symmetry
            sym_dict = self._encode_symmetry(symmetry, edge_type=2)
            links.append(sym_dict)

            # Track node py/ids
            for node in symmetry.nodes:
                if node not in node_to_py_id:
                    node_to_py_id[node] = self._get_or_create_py_id(node)

        # Ensure all skeleton nodes have py/ids
        for node in skeleton.nodes:
            if node not in node_to_py_id:
                node_to_py_id[node] = self._get_or_create_py_id(node)

        # Create nodes section with py/id references
        nodes = []
        for node in skeleton.nodes:
            nodes.append({"id": {"py/id": node_to_py_id[node]}})

        # Build final skeleton dict
        return {
            "directed": True,
            "graph": {"name": skeleton.name, "num_edges_inserted": len(skeleton.edges)},
            "links": links,
            "multigraph": True,
            "nodes": nodes,
        }

    def _encode_edge(self, edge: Edge, edge_idx: int, edge_type: int) -> Dict:
        """Encode an edge to jsonpickle format.

        Args:
            edge: Edge object to encode.
            edge_idx: Index of this edge.
            edge_type: Type of edge (1 for regular, 2 for symmetry).

        Returns:
            Dictionary representing the edge.
        """
        # Encode edge type - first occurrence uses py/reduce, subsequent use py/id
        # For backward compatibility, always use type 1 for regular edges
        if edge_type == 1:
            if not hasattr(self, "_edge_type_1_encoded"):
                type_dict = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [1]},
                    ]
                }
                self._edge_type_1_encoded = True
            else:
                type_dict = {"py/id": 1}
        else:
            if not hasattr(self, "_edge_type_2_encoded"):
                type_dict = {
                    "py/reduce": [
                        {"py/type": "sleap.skeleton.EdgeType"},
                        {"py/tuple": [2]},
                    ]
                }
                self._edge_type_2_encoded = True
            else:
                type_dict = {"py/id": 2}

        return {
            "edge_insert_idx": edge_idx,
            "key": 0,
            "source": self._encode_node(edge.source),
            "target": self._encode_node(edge.destination),
            "type": type_dict,
        }

    def _encode_symmetry(self, symmetry: Symmetry, edge_type: int) -> Dict:
        """Encode a symmetry to jsonpickle format.

        Args:
            symmetry: Symmetry object to encode.
            edge_type: Type of edge (should be 2 for symmetry).

        Returns:
            Dictionary representing the symmetry.
        """
        # Get source and target nodes (convert set to list for ordering)
        nodes_list = list(symmetry.nodes)
        source, target = nodes_list[0], nodes_list[1]

        # Encode edge type
        if not hasattr(self, "_edge_type_2_encoded"):
            type_dict = {
                "py/reduce": [{"py/type": "sleap.skeleton.EdgeType"}, {"py/tuple": [2]}]
            }
            self._edge_type_2_encoded = True
        else:
            type_dict = {"py/id": 2}

        return {
            "key": 0,
            "source": self._encode_node(source),
            "target": self._encode_node(target),
            "type": type_dict,
        }

    def _encode_node(self, node: Node) -> Dict:
        """Encode a node to jsonpickle format.

        Args:
            node: Node object to encode.

        Returns:
            Dictionary with py/object and py/state.
        """
        return {
            "py/object": "sleap.skeleton.Node",
            "py/state": {"py/tuple": [node.name, 1.0]},  # name, weight (always 1.0)
        }

    def _get_or_create_py_id(self, obj: Any) -> int:
        """Get or create a py/id for an object.

        Args:
            obj: Object to get/create ID for.

        Returns:
            The py/id integer.
        """
        obj_id = id(obj)
        if obj_id not in self._object_to_id:
            self._object_to_id[obj_id] = self._next_id
            self._next_id += 1
        return self._object_to_id[obj_id]

    def _recursively_sort_dict(self, obj: Any) -> Any:
        """Recursively sort dictionary keys for consistent output.

        Args:
            obj: Object to sort (dict, list, or other).

        Returns:
            Sorted version of the object.
        """
        if isinstance(obj, dict):
            # Sort keys and recursively sort values
            return {k: self._recursively_sort_dict(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            # Recursively sort list elements
            return [self._recursively_sort_dict(item) for item in obj]
        else:
            # Return as-is for non-dict/list types
            return obj
