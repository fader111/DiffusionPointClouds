import torch 
from torch_geometric.nn import ChebConv  # Using Chebyshev convolution
from torch_geometric.nn import knn_graph
from torch_geometric.utils import get_laplacian, get_mesh_laplacian
# from torch_sparse import coalesce
from torch_geometric.utils import degree

def compute_edge_indices(point_cloud, k=6):
    """
    Compute edge indices using k-nearest neighbors for the point cloud.
    
    Parameters:
        point_cloud (torch.Tensor): Tensor of shape [N, F], where N is the number of points and F is the feature dimension.
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
    """
    edge_index = knn_graph(point_cloud, k=k, loop=False)
    return edge_index

def compute_batched_edge_indices(X, k=6):
    """
    Compute edge indices for a batch of point clouds using k-nearest neighbors.
    
    Parameters:
        X (torch.Tensor): A tensor of shape [batch_size, num_points, num_features] representing the batched point cloud.
        k (int): Number of nearest neighbors to compute for each point.
        device (str): Device to move the tensors to (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Edge index tensor representing the k-NN graph for the entire batch.
    """
    batch_size, num_points, num_features = X.size()
    
    # Flatten the batch of point clouds into a single point cloud of shape [batch_size * num_points, num_features]
    X_flat = X.view(batch_size * num_points, num_features).to(X.device)
    
    # Create a batch index tensor that indicates which point belongs to which batch element
    batch_index = torch.arange(batch_size, device=X.device).repeat_interleave(num_points)
    
    # Compute the edge index for the entire batch using k-NN
    edge_index = knn_graph(X_flat, k=k, batch=batch_index, loop=False).to(X.device)
    
    return edge_index

def compute_laplacian(edge_index, num_nodes):
    """
    Compute the Laplacian matrix from the edge indices.
    
    Parameters:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
        num_nodes (int): Number of nodes in the point cloud.

    Returns:
        torch.Tensor: Laplacian matrix as a sparse tensor.
    """
    # Compute Laplacian using the same edge index
    # laplacian, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

    # NEW version - Get degrees for each node
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)

    # Compute normalized Laplacian weights (L = D - A)
    edge_weight = torch.ones((edge_index.size(1), ))  # Weight all edges equally
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Avoid division by zero

    # Apply normalization to the edge weights
    laplacian = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return laplacian

if __name__ == "__main__":
# Ensure the Laplacian is computed based on the correct edge index
# Recompute edge index for consistency
    point_cloud = torch.randn(1000, 3)  # Example point cloud with 1000 points
    edge_index = compute_edge_indices(point_cloud, k=6)
    print(f"Edge index shape: {edge_index.shape}")

    num_nodes = point_cloud.size(0)
    laplacian = compute_laplacian(edge_index, num_nodes)
    print(f"Laplacian shape: {laplacian.shape}")
