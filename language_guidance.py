import os
import sys
import numpy as np
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
from graspnetAPI.utils.eval_utils import compute_closest_points

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../f3rm'))

from f3rm.features.clip import clip
from f3rm.features.clip.clip import tokenize

'''
def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K = 50):
    
    **Input:**
    
    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format 
    
    - poses: from model to camera coordinate

    - config: dexnet config.
    
    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0/180*np.pi)

    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    pre_grasp_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp==i]
        grasp_i.sort_by_score()
        pre_grasp_list.append(grasp_i[:10].grasp_group_array)
    all_grasp_list = np.vstack(pre_grasp_list)
    remain_mask = np.argsort(all_grasp_list[:,0])[::-1]
    min_score = all_grasp_list[remain_mask[min(49,len(remain_mask) - 1)],0]

    grasp_list = []
    for i in range(num_models):
        remain_mask_i = pre_grasp_list[i][:,0] >= min_score
        grasp_list.append(pre_grasp_list[i][remain_mask_i])
    # grasp_list = pre_grasp_list

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)
    
    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()
    
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list
'''

def grasp_flitter(grasp_group, scene_path, top_indices, TOP_K = 50):

    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0/180*np.pi)

    scene = np.load(scene_path,allow_pickle=True).item()
    scene = np.array(scene['xyz'])
    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    gg = GraspGroup()
    for i,indice in enumerate(indices):
        if indice in top_indices:
            gg.add(grasp_group[i])
    return gg


def generate_lang_feat(obj_word_list):
    text_list = tokenize(obj_word_list)
    model, preprocess = clip.load("ViT-L/14@336px", device='cpu')
    text_features = model.encode_text(text_list)
    return text_features
    

def norm(t):
	return t / t.norm(dim=1, keepdim=True)

def cos_sim(v1, v2):
	v1 = norm(v1)
	v2 = norm(v2)
	return v1 @ v2.t()


def language_guidance(obj_word_list, scene_id, gg_path, vis=True):
    gg = GraspGroup()
    gg.from_npy(os.path.join(gg_path,"0000.npy"))
    
    fusion_path = "./data3/graspnet/fusion_scenes"
    feature_path = os.path.join(fusion_path,'scene_{}'.format(str(scene_id).zfill(4)),"realsense","clip_feature.npy")
    scene_path =  os.path.join(fusion_path,'scene_{}'.format(str(scene_id).zfill(4)),"realsense","points.npy")
    
    points_feature = np.load(feature_path)
    text_features = generate_lang_feat(obj_word_list)
    flitter_gg = GraspGroup()
    for i,text_feature in enumerate(text_features):
        similarity = cos_sim(text_feature.unsqueeze(0), torch.tensor(points_feature).float())
        _, top_indices = torch.topk(similarity, k=int(similarity.size(-1)*0.05))
        #print(similarity)
        #print(top_indices)
        top_indices = [i for i in top_indices[0] if similarity[0][i]>0.15]
        flitter_gg.add(grasp_flitter(gg, scene_path, top_indices))

    flitter_gg.save_npy(os.path.join(gg_path,"0000_{}.npy".format('_'.join(obj_word_list))))
    if vis:
        import copy
        gg = copy.deepcopy(flitter_gg)
        #gg.scores = gg.scores
        #scores = np.array(score_list)
        #scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
        #scores[collision_mask_list] = 0.3
        #gg.scores = scores
        gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
        grasps_geometry = gg.to_open3d_geometry_list()
        trans_ply_root = "../exports/pcd_scene_{}/point_cloud_trans.ply".format(str(scene_id).zfill(4))
        pcd = o3d.io.read_point_cloud(trans_ply_root)

        for grasp in grasps_geometry:
            pcd += grasp.sample_points_uniformly(number_of_points=2000)
        o3d.io.write_point_cloud("vis_grasp_{}_{}.ply".format(scene_id, '_'.join(obj_word_list)), pcd)

if __name__ == '__main__':
    obj_word_list = ["box"]
    scene_id = 108
    gg_path = "logs/log_dino_400/dump_epoch400_seen/scene_0108/realsense"
    language_guidance(obj_word_list, scene_id, gg_path)