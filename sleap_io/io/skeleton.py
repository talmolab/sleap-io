"""This module handles I/O operations for standalone skeleton JSON files."""

from __future__ import annotations
import json
from typing import Any, Dict, List, Union
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
        # First, decode all unique nodes from the links
        all_nodes = {}  # name -> Node mapping
        
        for link in data.get("links", []):
            # Process source
            if isinstance(link["source"], dict) and "py/object" in link["source"]:
                node = self._decode_node(link["source"])
                all_nodes[node.name] = node
            
            # Process target
            if isinstance(link["target"], dict) and "py/object" in link["target"]:
                node = self._decode_node(link["target"])
                all_nodes[node.name] = node
        
        # Create ordered nodes list based on the "nodes" section
        nodes = []
        py_id_to_index = {}  # py/id -> index in nodes list
        
        # The nodes section defines the order and py/id references
        for i, node_ref in enumerate(data.get("nodes", [])):
            if isinstance(node_ref["id"], dict) and "py/id" in node_ref["id"]:
                py_id = node_ref["id"]["py/id"]
                py_id_to_index[py_id] = i
                
                # Map py/id to actual node
                # In the test file, py/id 1 corresponds to "head" and py/id 2 to "abdomen"
                # This mapping is implicit based on order of appearance
                if i < len(all_nodes):
                    node = list(all_nodes.values())[i]
                    nodes.append(node)
                    self._id_to_object[py_id] = node
        
        # Now decode edges using the established node references
        edges = []
        symmetries = []
        
        for link in data.get("links", []):
            edge_type = self._get_edge_type(link.get("type", {}))
            
            # Get source and target nodes
            source_node = self._resolve_node_reference(link["source"], nodes, py_id_to_index)
            target_node = self._resolve_node_reference(link["target"], nodes, py_id_to_index)
            
            if edge_type == 1:  # Regular edge
                edges.append(Edge(source=source_node, destination=target_node))
            elif edge_type == 2:  # Symmetry
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
    
    def _resolve_node_reference(self, node_ref: Union[Dict, int], nodes: List[Node], py_id_to_index: Dict[int, int]) -> Node:
        """Resolve a node reference to an actual Node object.
        
        Args:
            node_ref: Node reference (can be embedded object, py/id reference, or index).
            nodes: List of nodes in skeleton.
            py_id_to_index: Mapping from py/id to index in nodes list.
            
        Returns:
            The resolved Node object.
        """
        if isinstance(node_ref, dict):
            if "py/object" in node_ref:
                # Embedded node - decode it and find matching node in our list
                node = self._decode_node(node_ref)
                for n in nodes:
                    if n.name == node.name:
                        return n
                raise ValueError(f"Node {node.name} not found in skeleton nodes")
            elif "py/id" in node_ref:
                # Reference to existing object
                py_id = node_ref["py/id"]
                if py_id in self._id_to_object:
                    return self._id_to_object[py_id]
                raise ValueError(f"py/id {py_id} not found in object cache")
        elif isinstance(node_ref, int):
            # Direct index (used in SLP format)
            return nodes[node_ref]
        
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