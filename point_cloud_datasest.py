import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from utils import compute_dense_laplacian, compute_normalized_laplacian, compute_manual_laplacian

# Custom dataset for point clouds
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, k=6):
        """
        Args:
            point_clouds: List of point cloud tensors, each of shape {num_points, point_dim}.
            k: Number of nearest neighbors for building the graph.
        """
        self.point_clouds = point_clouds
        self.k = k  # k-nearest neighbors

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        # Get the point cloud at the given index
        point_cloud = self.point_clouds[idx].reshape(-1, 3)  # shape {num_points, point_dim}
        num_points, point_dim = point_cloud.shape

        # Create edge_index using k-nearest neighbors (kNN)
        edge_index = knn_graph(point_cloud, k=self.k, loop=False)  # shape {2, num_edges}
        
        # Compute adjacency matrix from edge_index
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_points)[0]  # shape {num_points, num_points}

        # Compute the Laplacian (D - A) and return it as sparse representation
        # laplacian = compute_dense_laplacian(edge_index, num_points)
        # Or normalized 
        laplacian = compute_manual_laplacian(edge_index, num_points)
        # laplacian, _ = get_laplacian(edge_index, normalization="sym")


        return point_cloud, edge_index, laplacian

if __name__ == "__main__":
    
    # Example usage:
    point_clouds = [torch.rand((300)) for _ in range(64)]  # 1000 point clouds with 100 points each and 3D coordinates

    # Create dataset
    dataset = PointCloudDataset(point_clouds, k=6)

    # Create dataloader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example training loop
    for batch in train_loader:
        point_clouds, edge_indices, laplacians = batch
        # point_clouds: List of point clouds, shape {batch_size, num_points, point_dim}
        # edge_indices: List of edge_index for each point cloud
        # laplacians: List of laplacian tensors for each point cloud
        # Pass to your model for training
        # print(f"point_clouds, edge_indices, laplacians shapes\n{point_clouds.shape}, {edge_indices.shape}, {laplacians.shape}")
        print(f"{point_clouds.shape}, {edge_indices.shape}, {laplacians.shape}")