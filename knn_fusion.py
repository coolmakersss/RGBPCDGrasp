import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from f3rm.features.dinov2_extract import DINOv2Args, extract_dinov2_features
import torch
import torch.nn.functional as F
import cv2

def load_depth(depth_path, depth_scale=1000.0):
    """ 读取深度图并转换为米单位 """
    depth = cv2.imread(depth_path, -1).astype(np.float32)
    depth /= depth_scale  # 假设深度单位是毫米，需要转换为米
    return depth
'''
def resize_features(features, target_size):
    """ 将 DINOv2 特征图插值至深度图分辨率 """
    features = torch.tensor(features).permute(0, 3, 1, 2)  # (V, C, H, W)
    features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
    return features.permute(0, 2, 3, 1).numpy()  # 变回 (V, H, W, C)'
'''

def resize_features(features, target_size, device="cuda:0"):
    """ 逐 batch 上采样，减少显存占用 """
    V, H, W, C = features.shape
    features = features.cpu().numpy()
    features = torch.tensor(features, device=device).permute(0, 3, 1, 2)  # (V, C, H, W)
    
    batch_size = 1  # 逐批上采样，降低显存压力
    upscaled_features = []
    for i in range(0, V, batch_size):
        batch = features[i:i+batch_size]
        upscaled_batch = F.interpolate(batch, size=target_size, mode='bilinear', align_corners=False)
        upscaled_features.append(upscaled_batch)

    upscaled_features = torch.cat(upscaled_features, dim=0)
    return upscaled_features.permute(0, 2, 3, 1).cpu().numpy()  # (V, H, W, C)

def resize_features_cpu(features, target_size):
    """ 使用 OpenCV 逐帧上采样（CPU 低显存版本） """
    V, H, W, C = features.shape
    features = features.cpu().numpy()
    upscaled_features = np.zeros((V, *target_size, C), dtype=np.float32)
    
    for i in range(V):
        for j in range(C):  # 逐通道插值，降低内存占用
            upscaled_features[i, :, :, j] = cv2.resize(features[i, :, :, j], target_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    return upscaled_features

def project_features_to_world(depth, feature, K, T):
    """
    将单张RGB-D图像的特征投影到世界坐标系中。
    
    参数:
      depth: (H, W) 的深度图，每个像素的深度值（单位：米）
      feature: (H, W, C) 的特征图，每个像素对应一个 C 维的特征向量
      K: (3, 3) 相机内参矩阵
      T: (4, 4) 从相机坐标系到世界坐标系的外参变换矩阵
      
    返回:
      world_coords: (N, 3) 投影后有效像素的3D坐标（世界坐标系）
      feat: (N, C) 对应的特征向量
    """
    H, W = depth.shape
    # 生成像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    d = depth.flatten()
    feat = feature.reshape(-1, feature.shape[-1])
    
    # 过滤掉深度为0的无效点
    valid = (d > 0) & (d < 1.8)
    u = u[valid]
    v = v[valid]
    d = d[valid]
    feat = feat[valid]
    
    # 构造齐次坐标 [u, v, 1]
    pixels_homo = np.stack([u, v, np.ones_like(u)], axis=1)  # (N, 3)
    # 通过内参反投影到相机坐标系（未乘深度）
    K_inv = np.linalg.inv(K)
    cam_coords = (K_inv @ pixels_homo.T).T  # (N, 3)
    # 乘以深度获得正确的相机坐标
    cam_coords = cam_coords * d[:, np.newaxis]
    
    # 转换为齐次坐标
    cam_coords_homo = np.concatenate([cam_coords, np.ones((cam_coords.shape[0], 1))], axis=1)  # (N, 4)
    # 利用外参将点转换到世界坐标系
    #world_coords_homo = (T @ cam_coords_homo.T).T  # (N, 4)
    # 归一化
    world_coords = cam_coords_homo[:, :3] / cam_coords_homo[:, 3:4]
    
    return world_coords, feat

def aggregate_features(projected_points, projected_features, scene_points, k=3):
    """
    对场景点云中的每个点，根据投影的特征点进行最近邻搜索，并平均聚合邻域内的特征。
    
    参数:
      projected_points: (M, 3) 所有视角投影得到的3D点（世界坐标系）
      projected_features: (M, C) 与 projected_points 对应的特征
      scene_points: (N, 3) 场景点云中的点
      k: 最近邻的个数
      
    返回:
      agg_features: (N, C) 每个场景点聚合后的特征
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(projected_points)
    distances, indices = nbrs.kneighbors(scene_points)
    print(distances)
    
    # 对每个场景点，计算其k个邻居的特征均值
    agg_features = np.array([
        np.mean(projected_features[indices[i]], axis=0)
        for i in range(scene_points.shape[0])
    ])
    return agg_features

def main():
  for sceneId in range(139,187):
    root = "/home/xiangenda_2024/graspness_implementation/data3/graspnet/scenes/scene_{}/realsense".format(str(sceneId).zfill(4))
    camera_poses = np.load(os.path.join(root,"camera_poses.npy"))
    intrinsic_data = np.load(os.path.join(root,"camK.npy"))
    to_world_mat = np.load(os.path.join(root,"cam0_wrt_table.npy"))
    frames = []
    color_paths = []
    transforms = []
    depth_paths = []
    for i in range(0, 256, 32):
      color_path = os.path.join(root, "rgb", str(i).zfill(4) + ".png")
      depth_path = os.path.join(root, "depth", str(i).zfill(4) + ".png")
      # print(camera_poses[0])
      # exit()
      # print(to_world_mat)
      # exit()
      trans = np.dot(to_world_mat, camera_poses[i])
      transform_x = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
      trans = np.dot(trans,transform_x)
      # print(trans)
      # exit()
      # print(np.linalg.inv(camera_poses[3]))
      # exit()
      frame_data = {}
      frame_data["file_path"] = color_path
      color_paths.append(color_path)
      frame_data["depth_file_path"] = depth_path
      depth_paths.append(depth_path)
      frame_data["transform_matrix"] = [list(j) for j in trans]
      transforms.append([list(j) for j in trans])
      frames.append(frame_data)
    
    embeddings = extract_dinov2_features(color_paths,device=torch.device("cuda:0"))
    print(embeddings.shape) # views * 60 * 106 * 384
    embeddings = resize_features_cpu(embeddings, target_size=(720, 1280))  # 调整到 720x1280

    num_views = len(color_paths)
    projected_points_all = []
    projected_features_all = []
    
    # 以下为示例数据，请替换成你实际的数据
    for i in range(num_views):
      depth = load_depth(depth_paths[i])  # 读取深度
      feature = embeddings[i]  # 对应的 DINOv2 特征
      K = intrinsic_data  # 内参
      T = transforms[i]  # 外参

      pts, feats = project_features_to_world(depth, feature, K, T)
      print(len(pts))
      projected_points_all.append(pts)
      projected_features_all.append(feats)
    
    # 合并所有视角的投影点和特征
    projected_points_all = np.concatenate(projected_points_all, axis=0)
    projected_features_all = np.concatenate(projected_features_all, axis=0)
    print(len(projected_points_all))
    
  # 读取场景点云
    pcd_root = "/home/xiangenda_2024/graspness_implementation/data3/graspnet/fusion_scenes_dinov2_depth/scene_{}/realsense".format(str(sceneId).zfill(4))
    scene_points = np.load(os.path.join(pcd_root, "points.npy"),allow_pickle=True).item()  # 假设点云文件已存在
    scene_points = np.asarray(scene_points["xyz"])
    #print(len(scene_points))

    # 进行加权最近邻特征聚合
    agg_features = aggregate_features(projected_points_all, projected_features_all, scene_points, k=8)

    print("scene_{}聚合后的特征形状:".format(str(sceneId).zfill(4)), agg_features.shape)
    np.save(os.path.join(pcd_root,"agg_feature.npy"), agg_features)
    
if __name__ == "__main__":
    main()
