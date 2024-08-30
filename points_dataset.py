import os
import numpy as np
import plotly.graph_objects as go
import trimesh
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

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

def get_mesh_rt(filename): 
    "get quaternions and translation from json file"
    with open(filename, 'r') as f:
        # data = json.load(f)
        ...

def convert_to_head(mesh, filename):
    quaternion, translation = get_mesh_rt(filename)
    rotation = R.from_quat(quaternion).as_matrix()
    rotated_vertices = mesh.vertices @ rotation.T
    translated_vertices = rotated_vertices + np.array(translation)
    mesh.vertices = translated_vertices
    return mesh
    
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

    stl_dir =r"E:\awsCollectedDataPedro\stl"
    stl_dir =r"D:\Projects\webteeth-cra\public\meshes"
    dataset_dir = "datasets_embedded"
    points_per_shape = 256
    dataset_fname = f"ds_{points_per_shape}.pth"
    # root, dirs, files = load_stl_filenames(stl_dir)
    # print(f"root, dirs, files {root, dirs, files}")
    files = load_stl_filenames(stl_dir)
    data = []

    for filename in files:#[:10]: # debug 10 Items!!!!!!!!!!!!!
        mesh = load_mesh_fr_stl_file(filename)
        # convert meshes to head coordinates if needed
        mesh = convert_to_head(mesh, filename)

        points = get_points(mesh, points_per_shape)
        data.append(points)
    
    np_data = np.array(data, dtype=np.float32)
    # print(f"stls {files} {len(files)}")
    print(f"data {data[:3]}, np_data {np_data.shape}")
    ds = EmbedderDataset(np_data, points_per_shape)
    torch.save(ds, os.path.join(dataset_dir, dataset_fname))



