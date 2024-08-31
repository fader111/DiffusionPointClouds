import os
# import json
import numpy as np
import plotly.graph_objects as go
# import trimesh
import torch
from torch.utils.data import Dataset
from utils import *

# создаем датасет из stl файлов преобразуем mesh в облако точек, восстанавливаем его же
# id зуба не важен 
# 

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

    main_dir = r"E:\awsCollectedDataPedro"
    stl_dir = os.path.join(main_dir, "stl") 
    json_dir = os.path.join(main_dir, "collected_json_unmov_teeth")
    dataset_dir = "dataset_points"

    points_per_shape = 1024

    dataset_fname = f"ds_{points_per_shape}_head.pth"
    
    stl_folders = get_stl_folders(stl_dir) #[:3] # debug !!!!!!!!!!!!!!
    data = []

    for stl_folder in stl_folders:
        json_fname = os.path.join(json_dir, os.path.basename(stl_folder) + ".json")
        try:
            teeth_rt = get_teeth_rt(json_fname)
        except:
            print(f"json file not found {json_fname}")
            continue
        stl_filenames = load_stl_filenames(stl_folder)
        for stl_filename in stl_filenames:
            mesh = load_mesh_fr_stl_file(stl_filename)

            # convert meshes to head coordinates if needed
            tooth_id = os.path.splitext(os.path.basename(stl_filename))[0]
            mesh = convert_to_head(mesh, teeth_rt[tooth_id]) 

            points = get_points(mesh, points_per_shape)
            data.append(points)
    
    np_data = np.array(data, dtype=np.float32)
    # print(f"stls {files} {len(files)}")
    print(f"data {data[:3]}, np_data {np_data.shape}")
    ds = EmbedderDataset(np_data, points_per_shape)
    torch.save(ds, os.path.join(dataset_dir, dataset_fname))



