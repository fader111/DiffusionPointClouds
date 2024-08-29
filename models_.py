import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv  # Using Chebyshev convolution

class DiffusionNetLayer(nn.Module):
    def __init__(self, in_features, out_features, K=6):
        super(DiffusionNetLayer, self).__init__()
        # Chebyshev convolution approximates the diffusion on graphs
        self.conv = ChebConv(in_features, out_features, K)

    def forward(self, x, edge_index, laplacian):
        # Perform the convolution on the graph defined by edge_index
        x = self.conv(x, edge_index, laplacian)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_dim):
        super(Encoder, self).__init__()
        self.diffusion1 = DiffusionNetLayer(in_features, hidden_features)
        self.diffusion2 = DiffusionNetLayer(hidden_features, latent_dim)

    def forward(self, x, edge_index, laplacian):
        x = self.diffusion1(x, edge_index, laplacian)
        x = self.diffusion2(x, edge_index, laplacian)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_features, out_features):
        super(Decoder, self).__init__()
        self.diffusion1 = DiffusionNetLayer(latent_dim, hidden_features)
        self.diffusion2 = DiffusionNetLayer(hidden_features, out_features)

    def forward(self, x, edge_index, laplacian):
        x = self.diffusion1(x, edge_index, laplacian)
        x = self.diffusion2(x, edge_index, laplacian)
        return x

class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_dim):
        super(DiffusionNetAutoencoder, self).__init__()
        self.encoder = Encoder(in_features, hidden_features, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_features, in_features)

    def forward(self, x, edge_index, laplacian):
        # Encode to latent space
        latent = self.encoder(x, edge_index, laplacian)
        # Decode back to original space
        reconstructed = self.decoder(latent, edge_index, laplacian)
        return reconstructed

