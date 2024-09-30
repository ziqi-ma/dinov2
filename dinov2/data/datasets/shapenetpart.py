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
import glob
import h5py


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


class EvalShapeNetPart(data.Dataset):
    def __init__(self, class_choice, subset=False, apply_rotation=False):
        
        data_path = "/data/ziqi/shapenetpart"
        file = glob.glob(os.path.join(data_path, 'hdf5_data', '*test*.h5'))
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_seg = np.concatenate(all_seg, axis=0)
        all_rotation = torch.load(f"{data_path}/random_rotation_test.pt")

        self.data = all_data
        self.label = all_label
        self.seg = all_seg
        self.rotation = all_rotation
        self.apply_rotation = apply_rotation

        # misc.
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")

        self.cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
        
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

        id_choice = self.cat2id[class_choice]
        self.class_choice = class_choice
        indices = (self.label == id_choice).squeeze()
        self.data = self.data[indices]
        self.label = self.label[indices]
        self.seg = self.seg[indices]
        self.seg_num_all = self.seg_num[id_choice]
        self.seg_start_index = self.index_start[id_choice]
        self.rotation = self.rotation[indices]

        if subset:
            # get subset
            subset_idxs = np.loadtxt(f"/data/ziqi/shapenetpart/{class_choice}_subsample.txt").astype(int)
            self.data = self.data[subset_idxs]
            self.label = self.label[subset_idxs]
            self.seg = self.seg[subset_idxs]
            self.rotation = self.rotation[subset_idxs]

 
    def __getitem__(self, item):
        # subsample 5000 pts
        random_indices = torch.randint(0, self.data[item].shape[0], (5000,))
        pointcloud = self.data[item]#[random_indices]
        cat = self.class_choice
        gt = self.seg[item]- self.index_start[self.cat2id[cat]] + 1#[random_indices] - self.index_start[self.cat2id[cat]] + 1
        rot = self.rotation[item,:]

        ## encode label
        inputs = self.tokenizer(self.cat2part[cat], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        pts_xyz = torch.tensor(pointcloud).float()
        pts_rgb = torch.ones(pts_xyz.shape)*255 # no color
        normal = torch.tensor(np.asarray(pcd.normals))

        if self.apply_rotation:
            pts_xyz = rotate_pts(pts_xyz, rot)
        
        return_dict = prep_points_val3d(pts_xyz, pts_rgb, normal, gt)

        return_dict['label_embeds'] = text_feat # n_cur_mask, dim_feat, need to be padded
        return_dict['class_name'] = cat
        return_dict["xyz_full"] = pts_xyz
        return_dict["gt_full"] = gt

        return return_dict
    
    def __len__(self):
        return self.data.shape[0]