"""
Visualization utilities for computational graphs.

Provides functions to trace and visualize the computational graph
using graphviz.
"""

from graphviz import Digraph


def trace(root):
    """Trace the computational graph starting from a root node.
    
    Performs a depth-first traversal to find all nodes and edges
    in the computational graph.
    
    Args:
        root (Tensor): The root tensor (usually the loss).
        
    Returns:
        tuple: (topo, edges) where:
            - topo is a list of nodes in topological order
            - edges is a list of (source, dest) tuples
    """
    topo = []
    visited = set()
    edges = []

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                edges.append((child, v))
                build_topo(child)
            topo.append(v)

    build_topo(root)
    return topo, edges


def draw_dot(root):
    """Create a graphviz visualization of the computational graph.
    
    Generates a visual representation of the computational graph,
    showing each tensor's data and gradient values.
    
    Args:
        root (Tensor): The root tensor (usually the loss).
        
    Returns:
        Digraph: A graphviz Digraph object that can be rendered.
        
    Example:
        >>> dot = draw_dot(loss)
        >>> dot.render('graph', directory='/tmp', format='svg')
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    topo, edges = trace(root)
    for node in topo:
        uid = str(id(node))
        # Handle both scalars and arrays
        data_val = (
            float(node.data) if node.data.size == 1 else f"{node.data.shape}"
        )
        grad_val = (
            float(node.grad) if node.grad.size == 1 else f"{node.grad.shape}"
        )

        if isinstance(data_val, float):
            label = f"{{ data {data_val:.4f} | grad {grad_val:.4f} }}"
        else:
            label = f"{{ data {data_val} | grad {grad_val} }}"

        dot.node(name=uid, label=label, shape="record")

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    return dot
