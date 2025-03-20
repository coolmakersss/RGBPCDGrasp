import open3d as o3d
import numpy as np
import os
from graspnetAPI.utils.eval_utils import create_table_points, transform_points

if __name__ == '__main__':
    root_folder = "./"
    mesh_ply = o3d.geometry.TriangleMesh()
 
    for scene_id in range(100,189):
        # load ply
        mesh_ply = o3d.io.read_triangle_mesh(os.path.join(root_folder,"10_mesh", "{}.ply".format(str(scene_id).zfill(4))))
        mesh_ply.compute_vertex_normals()
    
        # V_mesh 为ply网格的顶点坐标序列，shape=(n,3)，这里n为此网格的顶点总数，其实就是浮点型的x,y,z三个浮点值组成的三维坐标
        V_mesh = np.asarray(mesh_ply.vertices)
        # F_mesh 为ply网格的面片序列，shape=(m,3)，这里m为此网格的三角面片总数，其实就是对顶点序号（下标）的一种组合，三个顶点组成一个三角形
        F_mesh = np.asarray(mesh_ply.triangles)
    
        print("ply info:", mesh_ply)
        print("ply vertices shape:", V_mesh.shape)
        print("ply triangles shape:", F_mesh.shape)
        #o3d.visualization.draw_geometries([mesh_ply], window_name="ply", mesh_show_wireframe=True)
    
    
        align_mat = np.load(os.path.join(root_folder,"data3/graspnet", 'scenes', 'scene_{}'.format(str(scene_id).zfill(4)), "realsense", 'cam0_wrt_table.npy'))
        camera_poses = np.load(os.path.join(root_folder,"data3/graspnet", 'scenes', 'scene_{}'.format(str(scene_id).zfill(4)), "realsense", 'camera_poses.npy'))
        camera_pose = camera_poses[0]
        camera_pose = np.matmul(align_mat, camera_pose)

        V_mesh = transform_points(V_mesh, np.linalg.inv(align_mat))
        
        # stl/ply -> pcd
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(V_mesh)
        #o3d.visualization.draw_geometries([pcd],window_name="pcd")
        # save pcd
        o3d.io.write_point_cloud(os.path.join(root_folder,"sample_pcd","{}.ply".format(str(scene_id).zfill(4))),pcd)

        # 保存为.npy文件
        data_dict = {'xyz': V_mesh,
                     'color': [np.array([0,0,0]) for x in V_mesh]}
        father_path = os.path.join(root_folder,"data3/graspnet","mhx",'scene_{}'.format(str(scene_id).zfill(4)),"realsense")
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        np.save(os.path.join(father_path,"points.npy"), data_dict)
        #np.save(os.path.join(father_path,"seg.npy"), data_seg)
