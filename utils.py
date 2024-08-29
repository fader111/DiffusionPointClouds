import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import degree

# Custom function to compute dense Laplacian
def compute_dense_laplacian(edge_index, num_nodes):
    """
    Compute the unnormalized Laplacian matrix L = D - A in dense format.
    
    Args:
        edge_index: Edge indices of the graph {2, num_edges}.
        num_nodes: Number of nodes in the graph.
    
    Returns:
        laplacian: Dense Laplacian matrix {num_nodes, num_nodes}.
    """
    # Compute the dense adjacency matrix A
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # {num_nodes, num_nodes}

    # Compute the degree matrix D
    degree_matrix = torch.diag(adj_matrix.sum(dim=1))  # {num_nodes, num_nodes}

    # Compute the Laplacian L = D - A
    laplacian = degree_matrix - adj_matrix

    return laplacian

def compute_normalized_laplacian(edge_index, num_nodes):
    """
    Compute the normalized Laplacian matrix Lsym = I - D^(-1/2) * A * D^(-1/2)
    """
    # Compute dense adjacency matrix A
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # {num_nodes, num_nodes}

    # Compute the degree matrix D
    degree_matrix = adj_matrix.sum(dim=1)  # {num_nodes}
    degree_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # Handle division by zero

    # Compute normalized Laplacian: I - D^(-1/2) A D^(-1/2)
    D_inv_sqrt = torch.diag(degree_inv_sqrt)  # {num_nodes, num_nodes}
    norm_laplacian = torch.eye(num_nodes) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return norm_laplacian

def compute_manual_laplacian(edge_index, num_nodes):
    """
    Manually compute the Laplacian matrix without changing the shape of edge_index.
    
    Parameters:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
        num_nodes (int): Number of nodes in the point cloud.

    Returns:
        torch.Tensor: Laplacian edge weights.
    """
    # Get degrees for each node
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)

    # Compute normalized Laplacian weights (L = D - A)
    edge_weight = torch.ones((edge_index.size(1), ), dtype=torch.float32)  # Weight all edges equally
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Avoid division by zero

    # Apply normalization to the edge weights
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return edge_weight

if __name__ == "__main__":

    # Example usage compute_dense_laplacian
    num_points = 100
    point_dim = 3
    point_cloud = torch.rand((num_points, point_dim))  # Point cloud with 100 points, 3D coordinates

    # Compute edge_index using k-NN graph
    edge_index = knn_graph(point_cloud, k=6, loop=False)  # {2, num_edges}

    # Compute the dense Laplacian matrix
    # laplacian = compute_dense_laplacian(edge_index, num_nodes=num_points)
    # or normalized
    # laplacian = compute_normalized_laplacian(edge_index, num_nodes=num_points)
    # or manually
    laplacian = compute_manual_laplacian(edge_index, num_nodes=num_points)
    print(laplacian.shape)  # Output: torch.Size([100, 100])