import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import ChebConv  # Using Chebyshev convolution

def autoencoder(dense_dim, dim_code):
    return nn.Sequential( # основной рабочий
            nn.Linear(dense_dim, dense_dim), # Это полносвязный слой
            nn.ELU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ELU(),
            nn.Linear(dense_dim, dim_code),

            nn.Linear(dim_code, dense_dim),
            nn.ELU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ELU(),
            nn.Linear(dense_dim, dense_dim)
            )
