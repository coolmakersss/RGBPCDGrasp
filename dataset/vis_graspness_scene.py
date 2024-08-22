import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
import numpy as np
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image

dataset_root = "../data3/graspnet"
scene_id = 10
ann_id = '0000'
camera_type = 'kinect'

#for scene_id in range(100):
fusion_data = np.load(os.path.join(dataset_root, 'fusion_scenes_dinov2', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'points.npy'),allow_pickle=True).item()
seg = np.load(os.path.join(dataset_root, 'fusion_scenes_dinov2', 'scene_' + str(scene_id).zfill(4), camera_type, 'seg.npy'))
point_cloud = np.array(fusion_data['xyz'])
color = np.ones_like(point_cloud)

graspness_full = np.load(os.path.join(dataset_root, 'graspness_dinov2', 'scene_' + str(scene_id).zfill(4), camera_type, 'graspness.npy')).squeeze()
graspness_full[seg == 0] = 0.
print('graspness full scene: ', graspness_full.shape, (graspness_full > 0.1).sum())
color[graspness_full > 0.1] = [0., 1., 0.]
#color[seg > 0] = [0., 1., 0.]


cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
#o3d.visualization.draw_geometries([cloud])
o3d.io.write_point_cloud("vis_graspness_cloud_scene.ply", cloud)
