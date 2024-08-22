import torch

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import tqdm


def to_o3d_pcd(points):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    return o3d_pcd

def get_color_mask(object_index, nc=None):
    object_index = object_index.astype(int)
    if nc is None:
        NUM_COLORS = len(np.unique(object_index))
    else:
        NUM_COLORS = nc
    print(NUM_COLORS)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i, index in enumerate(np.unique(object_index)):
        if index == 0 or index == -1:
            continue
        print(colors[i][:3])
        color_mask[object_index == index, :] = np.array(colors[i][:3])  * 255
        print(np.sum(object_index == index))
    return color_mask

root = "./data3/graspnet/"

for scene_id in tqdm.tqdm(range(0,6)):
    cad_path = os.path.join(root,"full_cad_scenes",'scene_{}'.format(str(scene_id).zfill(4)),"kinect")
    fusion_path = os.path.join(root, "fusion_scenes_fusion",'scene_{}'.format(str(scene_id).zfill(4)),"kinect")

    sim_pcd = np.array(np.load(os.path.join(cad_path,"points.npy")))
    sim_pcd = to_o3d_pcd(sim_pcd)
    sim_pcd_label = np.array(np.load(os.path.join(cad_path,"seg.npy")))
    # color_mask = get_color_mask(sim_pcd_label).astype(np.float32)
    # sim_pcd.colors = o3d.utility.Vector3dVector(color_mask)

    fusion_data = np.load(os.path.join(fusion_path,"points.npy"),allow_pickle=True).item()
    fusion_pcd = to_o3d_pcd(fusion_data['xyz'])
    fusion_pcd_label = np.zeros(np.array(fusion_data['xyz']).shape[0],)

    kdtree = o3d.geometry.KDTreeFlann(sim_pcd)
    tmp = 0
    for i, point in enumerate(fusion_pcd.points):
        [_, idx, _] = kdtree.search_hybrid_vector_3d(point, 0.01, 1)
        if idx:
            tmp+=1
            fusion_pcd_label[i] = sim_pcd_label[int(idx[0])]
    print(tmp)
    print(fusion_pcd_label.shape)
    print(len(fusion_data['xyz']))
    np.save(os.path.join(fusion_path,"seg.npy"),fusion_pcd_label)

# fusion_data = np.load("sim/points.npy",allow_pickle=True).item()
# point_cloud = np.array(fusion_data['xyz'])
# normal = np.array(fusion_data['normal'])
# print(len(point_cloud))
# # color = np.array(fusion_data['color'])
# seg = np.load("sim/seg.npy")
# pcd = o3d.geometry.PointCloud()
# print(np.unique(seg))
# pcd.points = o3d.utility.Vector3dVector(point_cloud[seg==74])
# pcd.normals = o3d.utility.Vector3dVector(normal[seg==74])



# frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
# box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-0.5,-0.5,-0.02]), max_bound = np.array([0.5,0.5,0.2]))

# fusion_data = np.load("realsense/points.npy",allow_pickle=True).item()
# point_cloud = np.array(fusion_data['xyz'])
# normal = np.array(fusion_data['normal'])
# print(len(point_cloud))
# # color = np.array(fusion_data['color'])
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(point_cloud)
# pcd2.normals = o3d.utility.Vector3dVector(normal)

# pcd.colors = o3d.utility.Vector3dVector(normal)
# o3d.io.write_point_cloud("scene.ply",pcd)
# color_mask = get_color_mask(fusion_pcd_label).astype(np.float32)
# fusion_pcd.colors = o3d.utility.Vector3dVector(color_mask)
# o3d.visualization.draw_geometries([sim_pcd])
# o3d.visualization.draw_geometries([fusion_pcd])
# o3d.visualization.draw_geometries([pcd])