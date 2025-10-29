import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(
    graph_indices: torch.Tensor,
    graph_weights: torch.Tensor,
    num_neurons: int,
    title: str,
    pruned_indices: torch.Tensor = None,
    new_indices: torch.Tensor = None
):
    """
    Generates and displays a visualization of the neural graph.
    Highlights pruned and newly added connections.

    Args:
        graph_indices (torch.Tensor): Indices of the current connections (2, num_connections).
        graph_weights (torch.Tensor): Weights of the current connections (num_connections,).
        num_neurons (int): Total number of neurons in the graph.
        title (str): Title for the plot.
        pruned_indices (torch.Tensor, optional): Indices of connections that were pruned.
        new_indices (torch.Tensor, optional): Indices of connections that were newly added.
    """
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(num_neurons))

    # Add current connections
    current_edges = []
    for i in range(graph_indices.shape[1]):
        u, v = graph_indices[0, i].item(), graph_indices[1, i].item()
        weight = graph_weights[i].item()
        current_edges.append((u, v, {'weight': weight, 'color': 'gray', 'alpha': 0.5}))
    G.add_edges_from(current_edges)

    # Highlight pruned connections (if provided)
    if pruned_indices is not None:
        for i in range(pruned_indices.shape[1]):
            u, v = pruned_indices[0, i].item(), pruned_indices[1, i].item()
            # Check if the edge exists in the graph before trying to access it
            if G.has_edge(u, v):
                G.edges[u, v]['color'] = 'red'
                G.edges[u, v]['alpha'] = 0.8
                G.edges[u, v]['style'] = 'dashed'

    # Highlight new connections (if provided)
    if new_indices is not None:
        for i in range(new_indices.shape[1]):
            u, v = new_indices[0, i].item(), new_indices[1, i].item()
            # Check if the edge exists in the graph before trying to access it
            if G.has_edge(u, v):
                G.edges[u, v]['color'] = 'green'
                G.edges[u, v]['alpha'] = 0.8
                G.edges[u, v]['style'] = 'solid'

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, iterations=50, seed=42) # Consistent layout

    # Get edge attributes for drawing
    edge_colors = [G.edges[u, v]['color'] for u, v in G.edges()]
    edge_alphas = [G.edges[u, v]['alpha'] for u, v in G.edges()]
    edge_styles = [G.edges[u, v].get('style', 'solid') for u, v in G.edges()]

    # Draw the graph
    nx.draw(
        G, pos, 
        with_labels=False, 
        node_color='lightblue', 
        node_size=50, 
        alpha=0.8, 
        edge_color=edge_colors, 
        width=0.5, 
        style=edge_styles
    )

    plt.title(title)
    plt.pause(0.001) # Small pause to allow plot to render
    plt.show(block=True) # Display plot and block execution until closed
