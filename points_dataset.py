import os
import numpy as np
import plotly.graph_objects as go
import trimesh
import torch
from torch.utils.data import Dataset

# создаем датасет из stl файлов преобразуем mesh в облако точек, восстанавливаем его же
# id зуба не важен 
# 

def load_mesh_fr_stl_file(stl_file_path):
    return trimesh.load_mesh(stl_file_path)

def get_points(mesh, count = 1024):
    points, var = trimesh.sample.sample_surface_even(mesh, count = count)
    return points

def load_stl_filenames(dir_path):
    files_only = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            files_only.append(os.path.join(root, file))
    return files_only

class EmbedderDataset(Dataset):
    def __init__(self, data, points_per_shape, point_dim=3):
        super(EmbedderDataset, self).__init__()
        self.point_dim = point_dim
        self.data = data.reshape(-1, points_per_shape * self.point_dim)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx][0])
        # y = torch.from_numpy(self.data[idx][1])
        return x
    

if __name__ == '__main__':
    # нужно существенно больше данных. 
    # и сохранять stl файлы не вариант. 
    # поэтому сохраняем stl временно, ( или может не временно а просто наделать все stl в педро обычной валидацией)
    # освободив предварительно место на диске... 
    # наверное лучше сохранить. не понятно сколько точек на зуб понадобится. может придется менять. 
    #
    # stl_dir =r"E:\awsCollectedDataPeydroNew_2024\stl"
    stl_dir =r"E:\awsCollectedDataPedro\stl"
    dataset_dir = "datasets_embedded"
    points_per_shape = 256
    dataset_fname = f"ds_{points_per_shape}.pth"
    # root, dirs, files = load_stl_filenames(stl_dir)
    # print(f"root, dirs, files {root, dirs, files}")
    files = load_stl_filenames(stl_dir)
    data = []

    for filename in files:#[:10]: # debug 10 Items!!!!!!!!!!!!!
        mesh = load_mesh_fr_stl_file(filename)
        points = get_points(mesh, points_per_shape)
        data.append(points)
    
    np_data = np.array(data, dtype=np.float32)
    # print(f"stls {files} {len(files)}")
    print(f"data {data[:3]}, np_data {np_data.shape}")
    ds = EmbedderDataset(np_data, points_per_shape)
    torch.save(ds, os.path.join(dataset_dir, dataset_fname))



