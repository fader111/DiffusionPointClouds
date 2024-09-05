''' датасет который в качестве X берет выход кодера из PointCloudAutoencoder - зубы одного кейса T1 a y - T2 '''
import os
import trimesh
import torch
from models import DiffusionNetAutoencoder
from torch.utils.data import Dataset
from point_cloud_dataset import PointCloudDataset
from torch_geometric.nn import knn_graph
# from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from utils import *

# Custom dataset for point clouds
# NOTE тут возможна ситуация когда кейс не надо брать в работу - мало стейджей, или больше 32 зубов в кейсе
# поэтому надо сначало внешней функцией (по аналогии в json_data_parser) получить данные а потом из них делать датасет.

def convert_to_head_batch(tooth_meshes, tooth_rigid_transforms):
    ''' каждый меш конвертим в ск головы'''
    for tooth_id in tooth_meshes:
        tooth_meshes[tooth_id] = convert_to_head(tooth_meshes[tooth_id], tooth_rigid_transforms[tooth_id])
    return tooth_meshes

def get_case_data(stl_folder, json_dir, encoder, decoder, num_points, k):
    
    stl_fnames = [os.path.join(stl_folder, fname) for fname in os.listdir(stl_folder) if fname.endswith(".stl")]
    if len(stl_fnames) >32: # если в кейсе больше 32 зубов, не берем в работу
        return

    # tooth_meshes = [trimesh.load_mesh(fname) for fname in stl_fnames]
    tooth_meshes = {os.path.basename(fname).split('.')[0]: trimesh.load_mesh(fname) for fname in stl_fnames}
    # move tooth meshes to head coordinates
    json_fname = os.path.join(json_dir, os.path.basename(stl_folder) + ".json")
    # tooth_ids = [os.path.splitext(os.path.basename(fname))[0] for fname in stl_fnames]
    if not os.path.exists(json_fname):
        print(f"json file not found {json_fname}, skipped")
        return
    stage_t2 = get_t2_stage(json_fname)

    output = []
    for stage in (0, stage_t2):
        
        tooth_rigid_transforms = get_teeth_rt(json_fname, stage=stage) # -> dict with keys tooth_id and values dict with keys rotation and translation

        tooth_meshes = {tooth_id:convert_to_head(tooth_meshes[tooth_id], tooth_rigid_transforms[tooth_id]) for tooth_id in tooth_meshes}
        # point_clouds = {tooth_id:trimesh.sample.sample_surface_even(mesh, count=num_points)[0] for tooth_id, mesh in tooth_meshes.items()}
        point_clouds = {}
        for tooth_id, mesh in tooth_meshes.items():
            point_clouds[tooth_id] = trimesh.sample.sample_surface(mesh, count=num_points)[0]
            if len(point_clouds[tooth_id]) < num_points:
                print(f"wrong point num for case {stl_folder}")
                return

        # тут надо сделать наборы облаков длиной 32 зуба чтобы отсутствующие зубы заменялись нулями в нужных позициях
        point_clouds_data = get_positioned_data(point_clouds, num_points=num_points) 

        # если get_positioned_data вернул None то в кейсе есть одновременно и примари и основной зуб, не берем
        if not point_clouds_data:
            print(f"no point clouds for {stl_folder}")
            return

        batches = [PointCloudDataset((torch.from_numpy(point_cloud_data),), k=k) for point_cloud_data in point_clouds_data]
        embeddings = [encoder(batch[0].x, batch[0].edge_index, batch[0].edge_weight).detach().cpu().numpy() for batch in batches]
        output.append(embeddings)

    return output


def get_dataset_data(main_dir, stl_dir, json_dir, encoder_states_file_path, decoder_states_file_path, k=6):
    
    stl_dir = os.path.join(main_dir, stl_dir)
    json_dir = os.path.join(main_dir, json_dir)

    POINT_DIM = 3
    hidden_features = 32
    latent_dim = 2
    num_points = 128#256

    autoencoder = DiffusionNetAutoencoder(POINT_DIM, hidden_features, latent_dim)
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    encoder.load_state_dict(torch.load(encoder_states_file_path))
    decoder.load_state_dict(torch.load(decoder_states_file_path))

    ds_data = []

    for i, stl_folder in enumerate(os.listdir(stl_dir)):
        if not os.path.isdir(os.path.join(stl_dir, stl_folder)):
            continue
        case_data = get_case_data(os.path.join(stl_dir, stl_folder), json_dir, encoder, decoder, num_points, k)
        if not case_data:
            continue
        ds_data.append(case_data)
        if i % 100 == 0:
            print(f"processed {i}/{len(os.listdir(stl_dir))} cases")

    return np.array(ds_data, dtype=np.float32)


class AlignerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx][0]), torch.from_numpy(self.data[idx][1])


if __name__ == "__main__":

    ds_data = get_dataset_data(
        main_dir="E:\\awsCollectedDataPedro",
        stl_dir="stl",
        json_dir="collected_json_unmov_teeth",
        encoder_states_file_path="models3322/encoder_494.pth",
        decoder_states_file_path="models3322/decoder_494.pth",
        k=6
    )

    ds = AlignerDataset(ds_data) 
    # ds_data - [case, case, ...] где case = [t1, t2] t1 = [зуб1, зуб2, ..., зуб32] зуб.shape[256,32]
    ds_folder = "datasets_align"
    # вычисления занимают много место в памяти вычисляем датасет по частям, 
    # сохраняем части и конкатенируем их.
    # torch.save(ds, os.path.join(ds_folder, "dataset_256.pth"))
    torch.save(ds, os.path.join(ds_folder, "dataset_128.pth"))
    print(f"ds length {len(ds)}")
