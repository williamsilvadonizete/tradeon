"""
Simple Graph implementation for knowledge management
"""

from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleGraph:
    """A simple graph implementation for managing knowledge and relationships."""
    
    def __init__(self):
        """Initialize an empty graph."""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}
        self.node_data: Dict[str, Dict] = {}
        
    def add_node(self, node_id: str, data: Optional[Dict] = None) -> None:
        """Add a node to the graph with optional data."""
        self.nodes.add(node_id)
        self.edges[node_id] = set()
        if data:
            self.node_data[node_id] = data
        else:
            self.node_data[node_id] = {}
            
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between two nodes."""
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)
            
        self.edges[from_node].add(to_node)
        
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges from the graph."""
        if node_id in self.nodes:
            self.nodes.remove(node_id)
            del self.edges[node_id]
            if node_id in self.node_data:
                del self.node_data[node_id]
                
            # Remove edges pointing to this node
            for edges in self.edges.values():
                edges.discard(node_id)
                
    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge between two nodes."""
        if from_node in self.edges:
            self.edges[from_node].discard(to_node)
            
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all neighbors of a node."""
        return self.edges.get(node_id, set())
    
    def get_node_data(self, node_id: str) -> Dict:
        """Get data associated with a node."""
        return self.node_data.get(node_id, {})
    
    def update_node_data(self, node_id: str, data: Dict) -> None:
        """Update data associated with a node."""
        if node_id in self.nodes:
            self.node_data[node_id].update(data)
            
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes
    
    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if an edge exists between two nodes."""
        return from_node in self.edges and to_node in self.edges[from_node]
    
    def get_all_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        return self.nodes.copy()
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph as (from_node, to_node) pairs."""
        edges = []
        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                edges.append((from_node, to_node))
        return edges
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.node_data.clear() 