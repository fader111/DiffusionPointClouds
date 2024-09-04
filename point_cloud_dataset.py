import torch
from torch.utils.data import Dataset
from torch_geometric.nn import knn_graph
# from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree

# Custom dataset for point clouds
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, k=6, point_dim=3, num_points=256):
        """
        Args:
            point_clouds: List of point cloud tensors, each of shape {num_points, point_dim}.
            k: Number of nearest neighbors for building the graph.
        """
        self.point_clouds = point_clouds
        self.k = k  # k-nearest neighbors
        self.point_dim = point_dim
        self.num_points = num_points

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        # Get the point cloud at the given index
        point_cloud = self.point_clouds[idx].reshape(-1, self.point_dim)  # shape {num_points, point_dim}
        
        # Create edge_index using k-nearest neighbors (kNN)
        edge_index = knn_graph(point_cloud, k=self.k, loop=False)  # shape: [2, num_edges]

        # Compute Laplacian (sparse form using edge weights)
        edge_weight = self.compute_edge_weight(edge_index, point_cloud.size(0))

        # Prepare data for PyTorch Geometric Data object
        data = Data(x=point_cloud, edge_index=edge_index, edge_weight=edge_weight)

        return data

    def compute_edge_weight(self, edge_index, num_nodes):
        """
        Compute edge weights corresponding to the Laplacian matrix in sparse form.
        
        Args:
            edge_index (Tensor): Edge indices for the graph.
            num_nodes (int): Number of nodes (points) in the graph.
        
        Returns:
            edge_weight (Tensor): Edge weights corresponding to the Laplacian.
        """
        # Compute degree for each node
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

        # Compute edge weights for Laplacian: L = D - A
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)  # Adjacency weights (1 for each edge)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Apply normalized Laplacian: D^{-0.5} A D^{-0.5}
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_weight

if __name__ == "__main__":
    
    # Example usage:
    point_clouds = [torch.rand((300)) for _ in range(64)]  # 1000 point clouds with 100 points each and 3D coordinates

    # Create dataset
    dataset = PointCloudDataset(point_clouds, k=6)

    # Create dataloader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example training loop
    for batch in train_loader:
        x = batch.x  # Node features for all graphs in the batch
        edge_index = batch.edge_index  # Edge indices for all graphs in the batch
        edge_weight = batch.edge_weight  # Edge weights (Laplacian) for all graphs in the batch
        print(f"{x.shape}, {edge_index.shape}, {edge_weight.shape}")
        # print(f"{x.shape}, {len(edge_index)}")