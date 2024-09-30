from .transform import *
import torch.utils.data as data
import os
from transformers import AutoTokenizer, AutoModel
from typing import Any, Callable, List, Optional, Set, Tuple
from torch.utils.data.dataloader import default_collate
import os
import torch
import open3d as o3d
import json


def prep_points_val3d(xyz, rgb, normal, gt):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "gt":gt}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', "gt"),
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict


class EvalPartNetE(data.Dataset):
    def __init__(self, category, subset=False, apply_rotation=False):
        
        ids = sorted(os.listdir(f"/data/ziqi/partnet-mobility/test/{category}"))
        if subset:
            with open(f"/data/ziqi/partnet-mobility/test/{category}/subsampled_ids.txt", 'r') as f:
                self.obj_path_list = f.read().splitlines()
        else:
            self.obj_path_list = [f"/data/ziqi/partnet-mobility/test/{category}/{id}" for id in ids if "txt" not in id]

        self.category = category
        with open(f"/data/ziqi/partnet-mobility/PartNetE_meta.json") as f:
            all_mapping = json.load(f)
        self.part_names = all_mapping[category]
        self.apply_rotation = apply_rotation

        # misc.
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")

        ## encode label
        inputs = self.tokenizer(self.part_names, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        self.text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
    
 
    def __getitem__(self, item):
        return_dict = {}
        file_path = self.obj_path_list[item]
        pcd = o3d.io.read_point_cloud(f"{file_path}/pc.ply")
        rot = torch.load(f"{file_path}/rand_rotation.pt")
        
        pts_xyz = torch.tensor(np.asarray(pcd.points)).float()
        pts_rgb = torch.tensor(np.asarray(pcd.colors))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normal = torch.tensor(np.asarray(pcd.normals))

        # subsample 5000 pts
        if self.apply_rotation:
            pts_xyz = rotate_pts(pts_xyz, rot)
        
        random_indices = torch.randint(0, pts_xyz.shape[0], (5000,))
        pts_xyz_subsampled = pts_xyz[random_indices]
        pts_rgb_subsampled = pts_rgb[random_indices]
        normal_subsampled = normal[random_indices]
        
        # this gt is for all points, not subsampled
        gt = torch.tensor(np.load(f"{file_path}/label.npy",allow_pickle=True).item()['semantic_seg'])+1 # we make it agree with objaverse, 0 is unlabeled and 1-k labeled
        gt_subsampled = gt[random_indices]

        return_dict = prep_points_val3d(pts_xyz_subsampled, pts_rgb_subsampled, normal_subsampled, gt_subsampled)

        return_dict["xyz_full"] = pts_xyz
        return_dict["gt_full"] = gt
        return_dict['label_embeds'] = self.text_feat # n_cur_mask, dim_feat, need to be padded
        return_dict['class_name'] = self.category

        return return_dict
    
    def __len__(self):
        return len(self.obj_path_list)