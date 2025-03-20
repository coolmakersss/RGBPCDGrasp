import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as scio
import open3d as o3d
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


def data_process():
    root = "./data3/graspnet"
    camera_type = "realsense"
    scene_id = 20

    # get valid points
    #depth_mask = (depth > 0)
    #camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    #align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    #trans = np.dot(align_mat, camera_poses[int(index)])
    #workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    #mask = (depth_mask & workspace_mask)

    '''
    for scene_id in tqdm(range(160)):
        fusion_data = np.load(os.path.join(root, 'full_cad_scenes_table', 'scene_' + str(scene_id).zfill(4), camera_type, 'points.npy'),allow_pickle=True)
        cloud = {'xyz':fusion_data}
        point_save_path = os.path.join(root, 'full_cad_scenes_table', 'scene_' + str(scene_id).zfill(4), camera_type, 'points.npy')
        np.save(point_save_path, cloud)
    '''
    fusion_data = np.load(os.path.join(root, 'full_cad_scenes_table', 'scene_' + str(scene_id).zfill(4), camera_type, 'points.npy'),allow_pickle=True).item()
    fusion_data = fusion_data['xyz']
    print(fusion_data.shape)
    cloud_masked =fusion_data

    # sample points random
    if len(cloud_masked) >= 30000:
        idxs = np.random.choice(len(cloud_masked), 30000, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), 30000 - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    point_sampled = cloud_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_sampled.astype(np.float32))
    o3d.io.write_point_cloud("merged_point_cloud.ply", cloud)

if __name__ == '__main__':
    data_process()