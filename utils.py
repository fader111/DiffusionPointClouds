import os
import json
import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial.transform import Rotation as R

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

def get_stl_folders(dir_path):
    dirs = []
    [dirs.append(os.path.join(dir_path, d)) for d in os.listdir(dir_path)] 
    return dirs

def get_json_file_data(json_fname):
    with open(json_fname, 'r') as f:
        data = json.load(f)
        return data

def get_teeth_rt(json_fname): 
    "get quaternions and translation from json file"
    json_file_data = get_json_file_data(json_fname)
    teeth_transforms = json_file_data["Staging"][0]["ToothTransforms"]
    return teeth_transforms

def convert_to_head(mesh, tooth_rt):
    quaternion, translation = list(tooth_rt["rotation"].values()), list(tooth_rt["translation"].values())
    rotation = R.from_quat(quaternion).as_matrix()
    rotated_vertices = mesh.vertices @ rotation.T
    translated_vertices = rotated_vertices + np.array(translation, dtype=np.float64)
    mesh.vertices = translated_vertices
    return mesh
