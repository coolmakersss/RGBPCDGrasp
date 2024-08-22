import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from knn.knn_modules import knn
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')


if __name__ == '__main__':
    cfgs = parser.parse_args()
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'graspness_dinov2')

    num_views, num_angles, num_depths = 300, 12, 4
    fric_coef_thresh = 0.8
    point_grasp_num = num_views * num_angles * num_depths
    for scene_id in range(100,187):
        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        labels = np.load(
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = []
        for j in range(len(labels)):
            collision_dump.append(labels['arr_{}'.format(j)])



        fusion_data = np.load(os.path.join(dataset_root, 'fusion_scenes_dinov2', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'points.npy'),allow_pickle=True).item()
        seg = np.load(os.path.join(dataset_root, 'fusion_scenes_dinov2', 'scene_' + str(scene_id).zfill(4), camera_type, 'seg.npy'))
        cloud = np.array(fusion_data['xyz'])

        # remove outlier and get objectness label
        #depth_mask = (depth > 0)
        depth_mask = (seg > 0)
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'camera_poses.npy'))
        #camera_pose = camera_poses[ann_id]
        camera_pose = camera_poses[0]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'cam0_wrt_table.npy'))
        #trans = np.dot(align_mat, camera_pose)
        trans = align_mat
        #workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=False, outlier=0.02)
        #mask = (depth_mask & workspace_mask)
        cloud_masked = cloud
        #objectness_label = seg[mask]

        # get scene object and grasp info
        scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'annotations', '0000.xml'))
        pose_vectors = scene_reader.getposevectorlist()
        obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
        grasp_labels = {}
        for i in obj_list:
            file = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
            grasp_labels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32),
                                file['scores'].astype(np.float32))

        grasp_points = []
        grasp_points_graspness = []
        for i, (obj_idx, trans_) in enumerate(zip(obj_list, pose_list)):
            sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
            collision = collision_dump[i]  # Npoints * num_views * num_angles * num_depths
            num_points = sampled_points.shape[0]

            valid_grasp_mask = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
            valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
            graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
            target_points = transform_points(sampled_points, trans_)
            target_points = transform_points(target_points, np.linalg.inv(camera_pose))  # fix bug
            grasp_points.append(target_points)
            grasp_points_graspness.append(graspness.reshape(num_points, 1))
        grasp_points = np.vstack(grasp_points)
        grasp_points_graspness = np.vstack(grasp_points_graspness)

        grasp_points = torch.from_numpy(grasp_points).cuda()
        grasp_points_graspness = torch.from_numpy(grasp_points_graspness).cuda()
        grasp_points = grasp_points.transpose(0, 1).contiguous().unsqueeze(0)

        masked_points_num = cloud_masked.shape[0]
        cloud_masked_graspness = np.zeros((masked_points_num, 1))
        part_num = int(masked_points_num / 10000)
        for i in range(1, part_num + 2):   # lack of cuda memory
            if i == part_num + 1:
                cloud_masked_partial = cloud_masked[10000 * part_num:]
                if len(cloud_masked_partial) == 0:
                    break
            else:
                cloud_masked_partial = cloud_masked[10000 * (i - 1):(i * 10000)]
            cloud_masked_partial = torch.from_numpy(cloud_masked_partial).cuda()
            cloud_masked_partial = cloud_masked_partial.transpose(0, 1).contiguous().unsqueeze(0)
            nn_inds = knn(grasp_points, cloud_masked_partial, k=1).squeeze() - 1
            cloud_masked_graspness[10000 * (i - 1):(i * 10000)] = torch.index_select(
                grasp_points_graspness, 0, nn_inds).cpu().numpy()

        max_graspness = np.max(cloud_masked_graspness)
        min_graspness = np.min(cloud_masked_graspness)
        cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / (max_graspness - min_graspness)

        print(len(cloud_masked_graspness)==len(cloud))
        print(cloud_masked_graspness)
        np.save(os.path.join(save_path, 'graspness.npy'), cloud_masked_graspness)
