import os
import json
import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial.transform import Rotation as R

up_teeth_nums16 = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22,
                23, 24, 25, 26, 27, 28]  # Jaw_id = 2 верхняя / по 16 зубов
dw_teeth_nums16 = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42,
                    43, 44, 45, 46, 47, 48]  # Jaw_id = 1 нижняя  / 16 зубов

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

def get_stl_folders(stl_dir_path):
    dirs = []
    [dirs.append(os.path.join(stl_dir_path, d)) for d in os.listdir(stl_dir_path)] 
    return dirs

def get_json_file_data(json_fname):
    with open(json_fname, 'r') as f:
        data = json.load(f)
        return data
    
def get_t2_stage(json_fname):
    json_file_data = get_json_file_data(json_fname)
    return int(json_file_data["T2Stage"])

def get_teeth_rt(json_fname, stage=0): 
    "get quaternions and translation from json file"
    json_file_data = get_json_file_data(json_fname)
    teeth_transforms = json_file_data["Staging"][stage]["ToothTransforms"]
    return teeth_transforms

def convert_to_head(mesh, tooth_rt):
    quaternion, translation = list(tooth_rt["rotation"].values()), list(tooth_rt["translation"].values())
    rotation = R.from_quat(quaternion).as_matrix()
    rotated_vertices = mesh.vertices @ rotation.T
    translated_vertices = rotated_vertices + np.array(translation, dtype=np.float64)
    mesh.vertices = translated_vertices
    return mesh

def load_json_data(json_path):
    with open(json_path, 'rb') as json_file: # 'rb' for support cirillic symbols from json
        try:
           data = json.load(json_file)
        except:
            return None
    return data

def quaternion(transform):
    return [
    transform["rotation"]["x"],
    transform["rotation"]["y"],
    transform["rotation"]["z"],
    transform["rotation"]["w"],
    ]

def translate(transform):
    return np.array([
    transform["translation"]["x"],
    transform["translation"]["y"],
    transform["translation"]["z"]
    ])

def get_transform_matrix(transform):
    quat = quaternion(transform)
    translation = translate(transform)
    # Преобразуем кватернион в матрицу поворота
    rotation_matrix = R.from_quat(quat).as_matrix() 
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def get_positioned_data(point_clouds, num_points=256):

    target = []
    for tooth_id in dw_teeth_nums16+up_teeth_nums16:
        actual_tooth_id = tooth_id+40 if str(tooth_id+40) in point_clouds else tooth_id           
        # если есть и примари и основной зуб возвращаемся
        if (str(tooth_id) in point_clouds and str(tooth_id+40) in point_clouds):
            print (f"primary and secondary tooth_id {tooth_id}")
            # logging.info(f'excluded {json_paths[case_idx]} : primary and secondary tooth_id {tooth_id} ')
            return
        if str(actual_tooth_id) in point_clouds:  
            target.append(point_clouds[str(actual_tooth_id)].astype(np.float32))
            
        else: 
            target.append(np.zeros((num_points, 3), dtype=np.float32))

    return target
    
        
