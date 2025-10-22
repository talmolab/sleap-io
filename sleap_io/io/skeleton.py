"""This module handles I/O operations for standalone skeleton JSON files."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import simplejson as json

from sleap_io.model.skeleton import Edge, Node, Skeleton, Symmetry


def decode_skeleton(data: Union[str, Dict]) -> Union[Skeleton, List[Skeleton]]:
    """Decode skeleton(s) from JSON data using the default decoder.

    Args:
        data: JSON string or pre-parsed dictionary containing skeleton data.

    Returns:
        A single Skeleton or list of Skeletons depending on input format.
    """
    decoder = SkeletonDecoder()
    return decoder.decode(data)


def encode_skeleton(skeletons: Union[Skeleton, List[Skeleton]]) -> str:
    """Encode skeleton(s) to JSON string using the default encoder.

    Args:
        skeletons: A single Skeleton or list of Skeletons to encode.

    Returns:
        JSON string in jsonpickle format.
    """
    encoder = SkeletonEncoder()
    return encoder.encode(skeletons)


def decode_yaml_skeleton(yaml_data: str) -> Union[Skeleton, List[Skeleton]]:
    """Decode skeleton(s) from YAML data.

    Args:
        yaml_data: YAML string containing skeleton data.

    Returns:
        A single Skeleton or list of Skeletons depending on input format.
    """
    decoder = SkeletonYAMLDecoder()
    return decoder.decode(yaml_data)


def encode_yaml_skeleton(skeletons: Union[Skeleton, List[Skeleton]]) -> str:
    """Encode skeleton(s) to YAML string.

    Args:
        skeletons: A single Skeleton or list of Skeletons to encode.

    Returns:
        YAML string with skeleton names as top-level keys.
    """
    encoder = SkeletonYAMLEncoder()
    return encoder.encode(skeletons)


def decode_training_config(data: dict) -> Union[Skeleton, List[Skeleton]]:
    """Decode skeleton(s) from training config data.

    Args:
        data: Dictionary containing training config with embedded skeletons.

    Returns:
        A single Skeleton or list of Skeletons from the training config.

    Raises:
        ValueError: If the data is not a valid training config format.
    """
    if isinstance(data, dict) and "data" in data:
        if "labels" in data["data"] and "skeletons" in data["data"]["labels"]:
            # This is a training config file with embedded skeletons
            decoder = SkeletonDecoder()
            return decoder.decode(data["data"]["labels"]["skeletons"])

    # If not a valid training config, raise an exception
    raise ValueError(
        "Invalid training config format. Expected dictionary with "
        "'data.labels.skeletons' structure."
    )


def load_skeleton_from_json(json_data: str) -> Union[Skeleton, List[Skeleton]]:
    """Load skeleton(s) from JSON data, with automatic training config detection.

    Args:
        json_data: JSON string that could be standalone skeleton or training config.

    Returns:
        A single Skeleton or list of Skeletons.
    """
    # Try to detect if this is a training config file
    try:
        data = json.loads(json_data)
        if isinstance(data, dict) and "data" in data:
            if "labels" in data["data"] and "skeletons" in data["data"]["labels"]:
                # This is a training config file with embedded skeletons
                return decode_training_config(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        # Not a training config or invalid JSON structure
        pass

    # Fall back to regular skeleton JSON decoding
    return decode_skeleton(json_data)


class SkeletonDecoder:
    """Decode skeleton data from jsonpickle-encoded format.

    This decoder handles the custom jsonpickle format used by SLEAP for
    standalone skeleton JSON files, which differs from the format used
    within .slp files.
    """

    def __init__(self):
        """Initialize the decoder."""
        self.decoded_objects: List[
            Any
        ] = []  # List of decoded objects indexed by py/id - 1

    def decode(self, data: Union[str, Dict]) -> Union[Skeleton, List[Skeleton]]:
        """Decode skeleton(s) from JSON data.

        Args:
            data: JSON string or pre-parsed dictionary containing skeleton data.

        Returns:
            A single Skeleton or list of Skeletons depending on input format.
        """
        if isinstance(data, str):
            data = json.loads(data)

        # Reset decoded objects list for each decode operation
        self.decoded_objects = []

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
        # Validate input data
        if data is None:
            raise ValueError("Skeleton data cannot be None")
        if not isinstance(data, dict):
            raise TypeError(f"Skeleton data must be a dictionary, got {type(data)}")

        # Reset decoded objects list for this skeleton
        self.decoded_objects = []

        # Track edge types separately for formats that use separate py/id spaces
        edge_type_ids = {}  # edge_type_value -> py/id
        next_edge_type_id = 1

        # First pass: decode all objects in order of appearance
        seen_nodes = set()  # Track node names we've already seen

        # Handle both direct format and nx_graph format
        if "nx_graph" in data:
            # nx_graph format (standalone skeleton files)
            links_data = data["nx_graph"].get("links", [])
            nodes_data = data["nx_graph"].get("nodes", [])
            graph_data = data["nx_graph"].get("graph", {})
        else:
            # Direct format (training config embedded skeletons)
            links_data = data.get("links", [])
            nodes_data = data.get("nodes", [])
            graph_data = data.get("graph", {})

        for link in links_data:
            # Check each component of the link for new objects
            for key in ["source", "target", "type"]:
                value = link.get(key, {})
                if isinstance(value, dict):
                    if "py/object" in value:
                        # New node object
                        node = self._decode_node(value)
                        if node.name not in seen_nodes:
                            self.decoded_objects.append(node)
                            seen_nodes.add(node.name)
                    elif "py/reduce" in value:
                        # New edge type
                        edge_type_val = value["py/reduce"][1]["py/tuple"][0]
                        self.decoded_objects.append(edge_type_val)
                        # Also track edge type IDs separately
                        if edge_type_val not in edge_type_ids:
                            edge_type_ids[edge_type_val] = next_edge_type_id
                            next_edge_type_id += 1
                    # py/id references are handled in second pass

        # Also process nodes that are directly defined in the nodes array
        # This is crucial for single-node skeletons with no edges
        for node_ref in nodes_data:
            if isinstance(node_ref.get("id"), dict) and "py/object" in node_ref["id"]:
                # New node object directly in nodes array
                node = self._decode_node(node_ref["id"])
                if node.name not in seen_nodes:
                    self.decoded_objects.append(node)
                    seen_nodes.add(node.name)

        # Store edge type mappings for second pass
        self._edge_type_ids = edge_type_ids

        # Second pass: build edges using the decoded objects
        edges = []
        symmetries = []
        seen_symmetries = set()

        for link in links_data:
            # Resolve references to build the edge
            source_node = self._resolve_link_ref(link["source"])
            target_node = self._resolve_link_ref(link["target"])
            edge_type_val = self._resolve_edge_type_ref(link.get("type", {}))

            if edge_type_val == 1:  # Regular edge
                edges.append(Edge(source=source_node, destination=target_node))
            elif edge_type_val == 2:  # Symmetry edge
                # Create a unique key for this symmetry pair (order-independent)
                sym_key = tuple(sorted([source_node.name, target_node.name]))
                if sym_key not in seen_symmetries:
                    symmetries.append(Symmetry([source_node, target_node]))
                    seen_symmetries.add(sym_key)

        # Build nodes list from the nodes section
        nodes = []
        nodes_from_refs = []

        # First collect nodes based on the nodes array
        for node_ref in nodes_data:
            if isinstance(node_ref["id"], dict) and "py/id" in node_ref["id"]:
                py_id = node_ref["id"]["py/id"]
                # Get node from decoded objects (py/id is 1-indexed)
                if py_id <= len(self.decoded_objects):
                    obj = self.decoded_objects[py_id - 1]
                    if isinstance(obj, Node):
                        nodes_from_refs.append(obj)

        # If we're missing nodes (due to malformed JSON), collect all Node objects
        all_nodes = [obj for obj in self.decoded_objects if isinstance(obj, Node)]

        if len(nodes_from_refs) < len(all_nodes):
            # The nodes array is incomplete or includes non-nodes
            # Use all nodes in their natural order
            nodes = all_nodes
        else:
            # Use the order from the nodes array
            nodes = nodes_from_refs

        # Get skeleton name
        name = graph_data.get("name", "Skeleton")

        return Skeleton(nodes=nodes, edges=edges, symmetries=symmetries, name=name)

    def _resolve_link_ref(self, node_ref: Union[Dict, int]) -> Node:
        """Resolve a node reference.

        Args:
            node_ref: Node reference (can be embedded object or py/id reference).

        Returns:
            The resolved Node object.
        """
        if isinstance(node_ref, dict):
            if "py/object" in node_ref:
                # Find the node in decoded objects by name
                node = self._decode_node(node_ref)
                for obj in self.decoded_objects:
                    if isinstance(obj, Node) and obj.name == node.name:
                        return obj
                raise ValueError(f"Node {node.name} not found in decoded objects")
            elif "py/id" in node_ref:
                # Reference to existing object
                py_id = node_ref["py/id"]
                if py_id <= len(self.decoded_objects):
                    obj = self.decoded_objects[py_id - 1]
                    if isinstance(obj, Node):
                        return obj
                    raise ValueError(f"py/id {py_id} is not a Node")
                raise ValueError(f"py/id {py_id} not found")
        elif isinstance(node_ref, int):
            # Direct index (used in SLP format, shouldn't happen in standalone)
            raise ValueError(f"Direct index reference not supported: {node_ref}")

        raise ValueError(f"Unknown node reference format: {node_ref}")

    def _resolve_edge_type_ref(self, type_data: Dict) -> int:
        """Resolve edge type reference.

        Args:
            type_data: Dictionary containing edge type data.

        Returns:
            Integer edge type (1 for regular edge, 2 for symmetry).
        """
        if "py/reduce" in type_data:
            # Return the value directly (already decoded in first pass)
            return type_data["py/reduce"][1]["py/tuple"][0]
        elif "py/id" in type_data:
            # Reference to existing edge type
            py_id = type_data["py/id"]

            # First try to find in decoded objects (training config format)
            if py_id <= len(self.decoded_objects):
                obj = self.decoded_objects[py_id - 1]
                if isinstance(obj, int):
                    return obj

            # If not found, check if this is a separate edge type ID space
            # (standalone skeleton format)
            for edge_val, edge_id in self._edge_type_ids.items():
                if edge_id == py_id:
                    return edge_val

            raise ValueError(f"py/id {py_id} not found as edge type")
        else:
            # Default to regular edge
            return 1

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


class SkeletonSLPDecoder:
    """Decode skeleton data from SLP format.

    This decoder handles the SLP format used within .slp files, which uses
    integer indices for node references instead of embedded node objects.
    """

    def decode(self, metadata: dict, node_names: list[str]) -> list[Skeleton]:
        """Decode skeletons from SLP metadata format.

        Args:
            metadata: The metadata dict from an SLP file containing skeletons.
            node_names: Global list of node names from the SLP file.

        Returns:
            List of Skeleton objects.
        """
        skeleton_objects = []

        for skel in metadata["skeletons"]:
            # Parse out the cattr-based serialization stuff from the skeleton links.
            if "nx_graph" in skel:
                # New format introduced in SLEAP v1.3.2
                # TODO: Do something with the "description" and "preview_image" keys?
                skel = skel["nx_graph"]
            # Process links with proper py/id resolution.
            # In jsonpickle format, py/reduce creates a new object and assigns it
            # an implicit py/id (1, 2, 3...). We track which py/id maps to which
            # edge type value as we encounter them.
            edge_type_map = {}  # py/id -> edge_type_value
            next_py_id = 1
            edge_inds, symmetry_inds = [], []

            for link in skel["links"]:
                if "py/reduce" in link["type"]:
                    # New edge type definition - extract value and assign py/id
                    edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
                    edge_type_map[next_py_id] = edge_type
                    next_py_id += 1
                elif "py/id" in link["type"]:
                    # Reference to previously defined edge type - look up the value
                    py_id = link["type"]["py/id"]
                    # Fallback to py_id value if not in map (for files where edge types
                    # are defined in a separate scope or use implicit numbering)
                    edge_type = edge_type_map.get(py_id, py_id)

                if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
                    edge_inds.append((link["source"], link["target"]))
                elif edge_type == 2:
                    symmetry_inds.append((link["source"], link["target"]))

            # Re-index correctly.
            skeleton_node_inds = [node["id"] for node in skel["nodes"]]
            sorted_node_names = [node_names[i] for i in skeleton_node_inds]

            # Create nodes.
            nodes = []
            for name in sorted_node_names:
                nodes.append(Node(name=name))

            # Create edges.
            edge_inds = [
                (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
                for s, d in edge_inds
            ]
            edges = []
            for edge in edge_inds:
                edges.append(Edge(source=nodes[edge[0]], destination=nodes[edge[1]]))

            # Create symmetries.
            symmetry_inds = [
                (skeleton_node_inds.index(s), skeleton_node_inds.index(d))
                for s, d in symmetry_inds
            ]

            # Deduplicate symmetries - legacy files may have duplicates
            # (one for each direction)
            seen_symmetries = set()
            symmetries = []
            for symmetry in symmetry_inds:
                # Create a unique key for this symmetry pair (order-independent)
                sym_key = tuple(sorted([symmetry[0], symmetry[1]]))
                if sym_key not in seen_symmetries:
                    symmetries.append(
                        Symmetry([nodes[symmetry[0]], nodes[symmetry[1]]])
                    )
                    seen_symmetries.add(sym_key)

            # Create the full skeleton.
            skel = Skeleton(
                nodes=nodes,
                edges=edges,
                symmetries=symmetries,
                name=skel["graph"]["name"],
            )
            skeleton_objects.append(skel)

        return skeleton_objects


class SkeletonSLPEncoder:
    """Encode skeleton data to SLP format.

    This encoder produces the SLP format used within .slp files, which uses
    integer indices for node references instead of embedded node objects.
    """

    def encode_skeletons(
        self, skeletons: list[Skeleton]
    ) -> tuple[list[dict], list[dict]]:
        """Serialize a list of Skeleton objects to SLP format.

        Args:
            skeletons: A list of Skeleton objects.

        Returns:
            A tuple of (skeletons_dicts, nodes_dicts).

            nodes_dicts is a list of dicts containing the nodes in all the skeletons.
            skeletons_dicts is a list of dicts containing the skeletons.
        """
        # Create global list of nodes with all nodes from all skeletons.
        nodes_dicts = []
        node_to_id = {}
        for skeleton in skeletons:
            for node in skeleton.nodes:
                if node not in node_to_id:
                    node_to_id[node] = len(node_to_id)
                    nodes_dicts.append({"name": node.name, "weight": 1.0})

        skeletons_dicts = []
        for skeleton in skeletons:
            # Build links dicts for normal edges.
            edges_dicts = []
            for edge_ind, edge in enumerate(skeleton.edges):
                if edge_ind == 0:
                    edge_type = {
                        "py/reduce": [
                            {"py/type": "sleap.skeleton.EdgeType"},
                            {"py/tuple": [1]},  # 1 = real edge, 2 = symmetry edge
                        ]
                    }
                else:
                    edge_type = {"py/id": 1}

                edges_dicts.append(
                    {
                        "edge_insert_idx": edge_ind,
                        "key": 0,  # Always 0.
                        "source": node_to_id[edge.source],
                        "target": node_to_id[edge.destination],
                        "type": edge_type,
                    }
                )

            # Build links dicts for symmetry edges.
            for symmetry_ind, symmetry in enumerate(skeleton.symmetries):
                if symmetry_ind == 0:
                    edge_type = {
                        "py/reduce": [
                            {"py/type": "sleap.skeleton.EdgeType"},
                            {"py/tuple": [2]},  # 1 = real edge, 2 = symmetry edge
                        ]
                    }
                else:
                    edge_type = {"py/id": 2}

                src, dst = tuple(symmetry.nodes)
                edges_dicts.append(
                    {
                        "key": 0,
                        "source": node_to_id[src],
                        "target": node_to_id[dst],
                        "type": edge_type,
                    }
                )

            # Create skeleton dict.
            skeletons_dicts.append(
                {
                    "directed": True,
                    "graph": {
                        "name": skeleton.name,
                        "num_edges_inserted": len(skeleton.edges),
                    },
                    "links": edges_dicts,
                    "multigraph": True,
                    "nodes": [{"id": node_to_id[node]} for node in skeleton.nodes],
                }
            )

        return skeletons_dicts, nodes_dicts


class SkeletonYAMLDecoder:
    """Decode skeleton data from simplified YAML format.

    This decoder handles a simplified YAML format that is more human-readable
    than the jsonpickle format.
    """

    def decode(self, data: Union[str, Dict]) -> Union[Skeleton, List[Skeleton]]:
        """Decode skeleton(s) from YAML data.

        Args:
            data: YAML string or pre-parsed dictionary containing skeleton data.
                  If a dict is provided with skeleton names as keys, returns list.
                  If a dict is provided with nodes/edges/symmetries, returns single
                  skeleton.

        Returns:
            A single Skeleton or list of Skeletons depending on input format.
        """
        if isinstance(data, str):
            import yaml

            data = yaml.safe_load(data)

        # Check if this is a single skeleton dict or multiple skeletons
        if isinstance(data, dict):
            # If it has nodes/edges keys, it's a single skeleton
            if "nodes" in data:
                return self._decode_skeleton(data)
            else:
                # Multiple skeletons with names as keys
                skeletons = []
                for name, skeleton_data in data.items():
                    skeleton = self._decode_skeleton(skeleton_data, name)
                    skeletons.append(skeleton)
                return skeletons

        raise ValueError(f"Unexpected data format: {type(data)}")

    def decode_dict(self, skeleton_data: Dict, name: str = "Skeleton") -> Skeleton:
        """Decode a single skeleton from a dictionary.

        This is useful when the skeleton data is embedded in a larger YAML structure.

        Args:
            skeleton_data: Dictionary containing nodes, edges, and symmetries.
            name: Name for the skeleton (default: "Skeleton").

        Returns:
            A Skeleton object.
        """
        return self._decode_skeleton(skeleton_data, name)

    def _decode_skeleton(self, data: Dict, name: Optional[str] = None) -> Skeleton:
        """Decode a single skeleton from dictionary data.

        Args:
            data: Dictionary containing skeleton data in simplified format.
            name: Optional name override for the skeleton.

        Returns:
            A Skeleton object.
        """
        # Create nodes
        nodes = []
        node_map = {}
        for node_data in data.get("nodes", []):
            node = Node(name=node_data["name"])
            nodes.append(node)
            node_map[node.name] = node

        # Create edges
        edges = []
        for edge_data in data.get("edges", []):
            source_name = edge_data["source"]["name"]
            dest_name = edge_data["destination"]["name"]
            edge = Edge(source=node_map[source_name], destination=node_map[dest_name])
            edges.append(edge)

        # Create symmetries
        symmetries = []
        for sym_data in data.get("symmetries", []):
            # Each symmetry is a list of 2 node specifications
            node1_name = sym_data[0]["name"]
            node2_name = sym_data[1]["name"]
            symmetry = Symmetry([node_map[node1_name], node_map[node2_name]])
            symmetries.append(symmetry)

        # Use provided name or get from data
        if name is None:
            name = data.get("name", "Skeleton")

        return Skeleton(nodes=nodes, edges=edges, symmetries=symmetries, name=name)


class SkeletonYAMLEncoder:
    """Encode skeleton data to simplified YAML format.

    This encoder produces a human-readable YAML format that is easier to
    edit manually than the jsonpickle format.
    """

    def encode(self, skeletons: Union[Skeleton, List[Skeleton]]) -> str:
        """Encode skeleton(s) to YAML string.

        Args:
            skeletons: A single Skeleton or list of Skeletons to encode.

        Returns:
            YAML string with skeleton names as top-level keys.
        """
        import yaml

        if isinstance(skeletons, Skeleton):
            skeletons = [skeletons]

        data = {}
        for skeleton in skeletons:
            skeleton_data = self.encode_dict(skeleton)
            data[skeleton.name] = skeleton_data

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def encode_dict(self, skeleton: Skeleton) -> Dict:
        """Encode a single skeleton to a dictionary.

        This is useful when embedding skeleton data in a larger YAML structure.

        Args:
            skeleton: Skeleton object to encode.

        Returns:
            Dictionary with nodes, edges, and symmetries.
        """
        # Encode nodes
        nodes = []
        for node in skeleton.nodes:
            nodes.append({"name": node.name})

        # Encode edges
        edges = []
        for edge in skeleton.edges:
            edges.append(
                {
                    "source": {"name": edge.source.name},
                    "destination": {"name": edge.destination.name},
                }
            )

        # Encode symmetries
        symmetries = []
        for symmetry in skeleton.symmetries:
            # Convert set to list and encode as pairs
            node_list = list(symmetry.nodes)
            symmetries.append(
                [{"name": node_list[0].name}, {"name": node_list[1].name}]
            )

        return {"nodes": nodes, "edges": edges, "symmetries": symmetries}
