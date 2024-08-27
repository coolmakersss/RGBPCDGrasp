""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask
from graspnetAPI.utils.utils import xmlReader,parse_posevector

class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'check':
            self.sceneIds = list(range(124,125))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict

    def get_data_label(self, index):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        #graspness = np.load(os.path.join(self.root, 'graspness', self.sceneIds[index], self.camera, 'graspness.npy'))
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        #graspness_sampled = graspness
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        print(obj_idxs)
        print(object_poses_list)
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list}
        return ret_dict


class GraspNetDataset_fusion(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=True, augment=False, load_label=True, voxel_size = 0.005, use_fine = False, exfeature = None):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.voxel_size = voxel_size
        self.exfeature = exfeature


        if split == 'train':
            self.sceneIds = list(range(0,100))
        if split == 'train_modify':
            self.sceneIds = list(range(0,100))
        if split == 'train_with_similar':
            self.sceneIds = list(range(0,100)) + list(range(130,160))
        elif split == 'test':
            self.sceneIds = list(range(131, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_seen_one':
            self.sceneIds = list(range(0, 30))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 187))
        elif split == 'train_novel':
            self.sceneIds = list(range(160, 190, 2))
        elif split == 'test_wos':
            self.sceneIds = list(range(100, 130)) + list(range(160, 190))
        elif split == 'check':
            self.sceneIds = list(range(124,125))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.pcdpath = []
        self.labelpath = []
        self.graspnesspath = []
        self.sampath = []
        self.scenename = []
        self.inspath = []
        self.featurepath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            if use_fine:
                self.pcdpath.append(os.path.join(root, 'full_cad_scenes_table', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'full_cad_scenes_table', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, 'cad_graspness.npy'))
            elif self.exfeature == 'dino':
                self.pcdpath.append(os.path.join(root, 'fusion_scenes_dino', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes_dino', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'graspness_dino', x, camera, 'graspness.npy'))
                self.featurepath.append(os.path.join(root, 'fusion_scenes_dino', x, camera, 'dino_feature.npy'))
            elif self.exfeature == 'clip':
                self.pcdpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, 'graspness.npy'))
                self.featurepath.append(os.path.join(root, 'fusion_scenes', x, camera, 'clip_feature.npy'))
            elif self.exfeature == 'dinov2':
                self.pcdpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, 'graspness.npy'))
                self.featurepath.append(os.path.join(root, 'fusion_scenes', x, camera, 'dinov2_feature.npy'))
            elif self.exfeature == 'mhx':
                self.pcdpath.append(os.path.join(root, 'mhx', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'mhx', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'mhx', x, camera, 'graspness.npy'))
                self.featurepath.append(os.path.join(root, 'mhx', x, camera, 'points.npy'))
            else:
                self.pcdpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'seg.npy'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, 'graspness.npy'))
                self.featurepath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
            self.inspath.append(os.path.join(root, 'insseg_realsense', x[6:]+'.npy'))
            self.sampath.append(os.path.join(root, 'sam_fusion', x[6:] + '.npy'))
            self.scenename.append(x.strip())
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.pcdpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        # if np.random.random() > 0.5:
        #     flip_mat = np.array([[-1, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, 1]])
        #     point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
        #     normals = transform_point_cloud(normals, flip_mat, '3x3')
        #     for i in range(len(object_poses_list)):
        #         object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
        #     aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        fusion_data = np.load(self.pcdpath[index],allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        color = np.array(fusion_data['color'])
        print(point_cloud.shape)
        print(color.shape)
        # seg = np.array(np.load(self.inspath[index]))
        # seg = np.array(np.load(self.sampath[index]))
        seg = np.array(np.load(self.labelpath[index]))
        if self.exfeature != None and self.exfeature != 'mhx':
            exfeature = np.array(np.load(self.featurepath[index]))
        else:
            exfeature = np.array(fusion_data['color'])
        scene = self.scenename[index]
        

        '''
        if self.camera == "kinect":
            mask_x = ((point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] <0.5))
            mask_y = ((point_cloud[:, 1] > -0.5) & (point_cloud[:, 1] < 0.5))
            mask_z = ((point_cloud[:, 2] > -0.02) & (point_cloud[:, 2] < 0.2))
            workspace_mask = (mask_x & mask_y & mask_z)
            point_cloud = point_cloud[workspace_mask]
            #normal = normal[workspace_mask]
            color = color[workspace_mask]
            seg = seg[workspace_mask]
            exfeature = exfeature[workspace_mask]
        '''

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[0])
            workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=False, outlier=0.02)
            mask = workspace_mask
            point_cloud = point_cloud[mask]
            color = color[mask]
            exfeature = exfeature[mask]
        
        if return_raw_cloud:
            return point_cloud
        # sample points
        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = point_cloud[idxs]
        color_sampled = color[idxs]
        seg_sampled = seg[idxs]
        exfeature_sampled = exfeature[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1
        

        align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
        scene_reader = xmlReader(
            os.path.join(self.root, 'scenes', scene, self.camera, 'annotations', '0000.xml'))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        object_poses_list = []
        for posevector in posevectors:
            obj_idx, pose = parse_posevector(posevector)
            pose = np.matmul(align_mat, pose)
            object_poses_list.append(pose[:3,:4])
            obj_list.append(obj_idx + 1)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['color'] = color_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['exfeature'] = exfeature_sampled
        return ret_dict

    def get_data_label(self, index):
        fusion_data = np.load(self.pcdpath[index], allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        #color = np.array(fusion_data['color'])
        seg = np.array(np.load(self.labelpath[index]))
        graspness = np.load(self.graspnesspath[index])
        if self.exfeature != None:
            exfeature = np.array(np.load(self.featurepath[index]))
        else:
            exfeature = np.array(fusion_data['color'])
        scene = self.scenename[index]

        '''
        if self.camera == "kinect":
            mask_x = ((point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] <0.5))
            mask_y = ((point_cloud[:, 1] > -0.5) & (point_cloud[:, 1] < 0.5))
            mask_z = ((point_cloud[:, 2] > -0.02) & (point_cloud[:, 2] < 0.2))
            workspace_mask = (mask_x & mask_y & mask_z)
            point_cloud = point_cloud[workspace_mask]
            #normal = normal[workspace_mask]
            #color = color[workspace_mask]
            seg = seg[workspace_mask]
            graspness = graspness[workspace_mask]
            exfeature = exfeature[workspace_mask]
        '''


        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[0])
            workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=False, outlier=0.02)
            mask = workspace_mask

            point_cloud = point_cloud[mask]
            #color = color[mask]
            seg = seg[mask]
            graspness = graspness[mask]
            exfeature = exfeature[mask]

        # sample points
        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = point_cloud[idxs]
        #color_sampled = color[idxs]
        seg_sampled = seg[idxs]
        graspness_sampled = graspness[idxs]
        exfeature_sampled = exfeature[idxs]
        #print(seg_sampled)
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1


        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        ret_obj_list = []

        # get object poses
        align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
        scene_reader = xmlReader(
            os.path.join(self.root, 'scenes', scene, self.camera, 'annotations', '0000.xml'))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        poses = []
        for posevector in posevectors:
            obj_idx, pose = parse_posevector(posevector)
            #pose = np.matmul(align_mat,pose)
            poses.append(pose)
            obj_list.append(obj_idx+1)
        poses = np.asarray(poses).astype(np.float32)

        print(obj_list)
        for i, obj_idx in enumerate(obj_list):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[i, :3, :4])
            points, offsets, scores= self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)


            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])

            ret_obj_list.append(np.asarray([obj_idx - 1]).astype(np.int64))

            scores = scores[idxs].copy()
            collision = collision[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            #tolerance = tolerance[idxs].copy()
            #tolerance[collision] = 0
            #grasp_tolerance_list.append(tolerance)
        print(object_poses_list)



        #print(cloud_sampled.shape)
        #print(exfeature_sampled.shape)
        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        # ret_dict['feats'] = np.concatenate([cloud_sampled.astype(np.float32),color_sampled.astype(np.float32)],axis=1)
        #ret_dict['feats'] = normal_sampled.astype(np.float32)
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        #ret_dict['pcd_color'] = color_sampled.astype(np.float32)
        ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_widths_list'] = grasp_offsets_list
        ret_dict['grasp_scores_list'] = grasp_scores_list
        ret_dict['exfeature'] = exfeature_sampled.astype(np.float32)
        #ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        #ret_dict['instance_mask'] = seg_sampled
        #ret_dict['obj_list'] = ret_obj_list #np.asarray(obj_list).astype(np.int64)
        return ret_dict


def load_grasp_labels_ann(root):
    obj_names = list(range(1, 89))
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels

def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        valid_obj_idxs.append(i + 1)  # here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32)[..., 2],
                               label['scores'].astype(np.float32))

    return valid_obj_idxs, grasp_labels


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
