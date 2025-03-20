import os
import sys
import numpy as np
import argparse
import time
import torch
from PIL import Image
import scipy.io as scio
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image
import collections.abc as container_abcs
import MinkowskiEngine as ME

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, GraspNet_rgb, pred_decode
from dataset.graspnet_dataset import GraspNetDataset_fusion, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
#parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)
    return res

net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
#net = GraspNet_rgb(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

batch_interval = 1
net.eval()

depthpath = "./0000.png"
metapath = "./0000.mat"
depth = np.array(Image.open(depthpath))
meta = scio.loadmat(metapath)
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']

camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
# generate cloud
cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
depth_mask = (depth > 0) 
cloud_masked = cloud[depth_mask]

print(len(cloud_masked))
#cloud_masked = np.expand_dims(cloud_masked, axis=0) 
#cloud_masked = torch.from_numpy(cloud_masked).float().unsqueeze(0)
print(cloud_masked.shape)
batch_data = [{'point_clouds': cloud_masked.astype(np.float32),
        'coors': cloud_masked.astype(np.float32) / cfgs.voxel_size,
        'feats': np.ones_like(cloud_masked).astype(np.float32),
    }]
batch_data = minkowski_collate_fn(batch_data)

for key in batch_data:
    batch_data[key] = batch_data[key].to(device)
end_points = net(batch_data)