# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from .transform import *
import torch.utils.data as data
import os
from transformers import AutoTokenizer, AutoModel
from typing import Any, Callable, List, Optional, Set, Tuple
from torch.utils.data.dataloader import default_collate
import os
import torch
import open3d as o3d

# the ultimate data structure is as follows:
# note that the way to match student and teacher globally (with 2 versions of aug)
# is that the prediction passes through a postprocessing that chunks into 2 and
# re-cats for the teacher output (see ssl_meta_arch.py line 162-164 in get_teacher_output()),
# for a batch size of 3 with objects a,b, c the dataloader processes it as a sequence
# a_1,b_1,c_1,a_2,b_2,c_2 where _1 and _2 are two augmentations of each object
# the student output is s(a_1), s(b_1), s(c_1), s(a_2), s(b_2), s(c_2)
# teacher output is t(a_1), t(b_1), t(c_1), t(a_2), t(b_2), t(c_2), after chunking and cat
# teacher output is changed to t(a_2), t(b_2), t(c_2), t(a_1), t(b_1), t(c_1)
# this is then taken to dinoloss with the student output sequence, so s(a_1) corresponds to
# t(a_2)
# local augs are ordered as
# obj1 crop1,...obj64 crop1, obj1 crop2, ... obj 64 crop 2, ...
'''
data_dict = {
    'coord',
    'grid_coord',
    'offset',
    'color',
    'normal'
    where everything is the first augmentation of whole batch followed by
    the second augmentation of whole batch
    'local_crops':{
       'coord',
       'grid_coord',
       'offset',
       'color',
       'normal'
       where everything is whole batch crop 1 followed by whole batch crop 2 etc.
    }


}
'''


def global_aug(xyz, rgb, normal):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    # x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = RandomDropout(dropout_ratio=0.2,dropout_application_ratio=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='z',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.05)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.05)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    data_dict = SphereCrop(sample_rate=0.8, mode='random')(data_dict)
    #data_dict = SphereCrop(point_max=204800, mode='random')(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    #data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord'),
                        offset_keys_dict={"offset":"coord"},
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict


def local_aug(xyz, rgb, normal):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    # x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = RandomDropout(dropout_ratio=0.4,dropout_application_ratio=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='z',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.05)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.05)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    data_dict = SphereCrop(sample_rate=0.6, mode='random')(data_dict)
    #data_dict = SphereCrop(point_max=204800, mode='random')(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    #data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord'),
                        offset_keys_dict={"offset":"coord"},
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict


def prep_points_val(xyz, rgb, normal):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal}
    data_dict = CenterShift(apply_z=True)(data_dict)
    
    # try global rotate
    #data_dict = RandomRotate(angle=[rotz, rotz],axis='z',p=1)(data_dict)
    #data_dict = RandomRotate(angle=[rotx, rotx],axis='x',p=1)(data_dict)
    #data_dict = RandomRotate(angle=[roty, roty],axis='y',p=1)(data_dict)
    # try rotate
    '''
    data_dict = RandomRotate(angle=[-1, 1],axis='z',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    '''

    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    #data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord'),
                        offset_keys_dict={"offset":"coord"},
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict

def prep_points_finetune(xyz, rgb, normal, mask2pt):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    # x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "mask2pt": mask2pt}
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = RandomDropout(dropout_ratio=0.2,dropout_application_ratio=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='z',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.15)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.15)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    #data_dict = SphereCrop(sample_rate=0.4, mode='random')(data_dict)
    #data_dict = SphereCrop(point_max=204800, mode='random')(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    #data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', 'mask2pt'),
                        offset_keys_dict={"offset":"coord"},
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict


def prep_points_finetune_val(xyz, rgb, normal, mask2pt, pt2face):
    # xyz, rgb, normal all (n,3) numpy arrays
    # rgb is 0-255
    # first shift coordinate frame since model is trained on depth coordinate
    # x revert, y z shift
    xyz_change_axis = np.concatenate([-xyz[:,0].reshape(-1,1), xyz[:,2].reshape(-1,1), xyz[:,1].reshape(-1,1)], axis=1)
    data_dict = {"coord": xyz_change_axis, "color": rgb, "normal":normal, "mask2pt": mask2pt, 'point2face': pt2face}
    data_dict = CenterShift(apply_z=True)(data_dict)
    '''
    data_dict = RandomRotate(angle=[-1, 1],axis='z',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-1, 1],axis='y',p=1)(data_dict)
    '''
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord', 'mask2pt', 'point2face'),
                        offset_keys_dict={"offset":"coord"},
                        feat_keys=('color', 'normal'))(data_dict)
    return data_dict


def visualize_pts(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    o3d.visualization.draw_plotly([pcd])

def visualize_data(data_dict, idx_list):
    # data_dict for a batch
    idx_offsets = data_dict["offset"]
    obj_start_idxs = torch.cat((torch.tensor([0]), idx_offsets[:-1]))
    for query_idx in idx_list:
        idx_start = obj_start_idxs[query_idx]
        idx_end = obj_start_idxs[query_idx+1]
        curobj_coords = data_dict["coord"][idx_start:idx_end,:]
        curobj_rgb = (data_dict["feat"][idx_start:idx_end,:3]+1)*127.5
        visualize_pts(curobj_coords, curobj_rgb)

def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    
    # first, order the two augmentations for teacher and student
    # note now batch is list of dict
    # [
        #{global_aug1: ..., global_aug2:..., local_augs:[list]}
        #{global_aug1: ..., global_aug2:..., local_augs:[list]}
        #...
    # ]
    # first reorder into one big sequence of 
    # [item 1 aug1, item 2 aug 1, ... item 1 aug2, item 2 aug2, ...]

    if isinstance(batch[0], Mapping) and "global_aug1" in batch[0] and "global_aug2" in batch[0]:
        global_flattened_batch = [item["global_aug1"] for item in batch] + [item["global_aug2"] for item in batch]
        global_collated_dict = collate_fn(global_flattened_batch) # this is one dict with each key corresponding to collated global data
        # add collated local into a key called local_augs
        # NOTE we need every object's crop1 followed by every object's crop2 etc.
        if "local_augs" in batch[0]:
            n_local_crops = len(batch[0]["local_augs"])
            local_flattened_batch = []
            for icrop in range(n_local_crops):
                for item in batch:
                    local_flattened_batch.append(item["local_augs"][icrop])
            # this is in the order of obj1crop1, ...obj64 crop1, obj1crop2, ...obj64crop2 etc.
            local_collated_dict = collate_fn(local_flattened_batch)
            global_collated_dict["local_crops"] = local_collated_dict
        return global_collated_dict
    if isinstance(batch[0], torch.Tensor):
        if len(batch)>1:
            if batch[0].shape[1:] == batch[1].shape[1:]:
                return torch.cat(list(batch))
            else:
                return list(batch) # not uniform shape
        else: # only one item, e.g. mask2pt, return itself
            return batch[0]
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        if isinstance(batch[0][0], str):
            return batch
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        # if there is mask2pt, always just append as a list
        batch_new = {key: collate_fn([d[key] for d in batch]) for key in batch[0] if key != "mask2pt"}
        if "mask2pt" in batch[0]:
            collated_mask2pt =  [d["mask2pt"] for d in batch]
            batch_new["mask2pt"] = collated_mask2pt
        for key in batch_new.keys():
            if "offset" in key:
                batch_new[key] = torch.cumsum(batch_new[key], dim=0)
        return batch_new
    else:
        return default_collate(batch)


class ObjaverseAugmented(data.Dataset):
    def __init__(
        self,
        *,
        n_local_crops:int,
        split: str, # train/test
        root: str
    ) -> None:
        self.dirpath = f"{root}/{split}"
        self.objs = os.listdir(self.dirpath)
        self.n_local_crops = n_local_crops
        print(f"{self.n_local_crops} local crops")

    def __getitem__(self, index: int) -> dict:
        cur_obj = self.objs[index]
        obj_dir = f"{self.dirpath}/{cur_obj}"
        pts_xyz = torch.load(f"{obj_dir}/points.pt")
        normal = torch.load(f"{obj_dir}/normals.pt")
        pts_rgb = torch.load(f"{obj_dir}/rgb.pt")*255 #torch.ones(normal.shape)*127.5 #
        point_global_aug1 = global_aug(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy())
        point_global_aug1["path"] = obj_dir
        point_global_aug2 = global_aug(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy())
        point_global_aug2["path"] = obj_dir

        # get 8 local crops:
        local_crops_list = []
        for i in range(self.n_local_crops):
            local_crops_list.append(local_aug(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy()))

        return {"global_aug1":point_global_aug1, "global_aug2":point_global_aug2, "local_augs":local_crops_list}

    def __len__(self) -> int:
        return len(self.objs)
    

class ObjaverseEval(data.Dataset):
    def __init__(
        self,
        *,
        split: str, # train/test
        root: str
    ) -> None:
        self.dirpath = f"{root}/{split}"
        self.objs = os.listdir(self.dirpath)
        self.label2idx = {}
        i = 0
        for obj in self.objs:
            label = ("_").join(obj.split("_")[:-1])
            if label not in self.label2idx:
                self.label2idx[label] = i
                i += 1

    def __getitem__(self, index: int) -> dict:
        cur_obj = self.objs[index]
        obj_dir = f"{self.dirpath}/{cur_obj}"
        pts_xyz = torch.load(f"{obj_dir}/points.pt")
        normal = torch.load(f"{obj_dir}/normals.pt")
        pts_rgb = torch.load(f"{obj_dir}/rgb.pt")*255 #torch.ones(normal.shape)*127.5
        point_dict = prep_points_val(pts_xyz, pts_rgb, normal)
        point_dict["label"] = self.label2idx[("_").join(cur_obj.split("_")[:-1])]
        point_dict["index"] = index
        point_dict["path"] = obj_dir
        return point_dict

    def __len__(self) -> int:
        return len(self.objs)
    

class ObjaverseEvalSubset(data.Dataset):
    def __init__(
        self,
        *,
        split: str, # train/test/val
        root: str,
        test_subset_idxs: list
    ) -> None:
        self.dirpath = f"{root}/{split}"
        #self.objs = [os.listdir(self.dirpath)[idx] for idx in test_subset_idxs]
        self.objs = [
            "chair_781e14252fd745bab52da09998441f5b",
            "cone_362cbe01dc734766927b2f4b50547a58",
            "mug_057ed726c32c4e0d8912d04343e7bf5a",
            "mushroom_2dd809a046d443a68daf28b50fb994f7",
            "race_car_3f31920c10a546b3955ba311b9c6ded6"
        ]
        self.label2idx = {}
        i = 0
        for obj in self.objs:
            label = ("_").join(obj.split("_")[:-1])
            if label not in self.label2idx:
                self.label2idx[label] = i
                i += 1

    def __getitem__(self, index: int) -> dict:
        cur_obj = self.objs[index]
        obj_dir = f"{self.dirpath}/{cur_obj}"
        pts_xyz = torch.load(f"{obj_dir}/points.pt")
        normal = torch.load(f"{obj_dir}/normals.pt")
        pts_rgb = torch.load(f"{obj_dir}/rgb.pt")*255
        point_dict = prep_points_val(pts_xyz, pts_rgb, normal)
        point_dict["label"] = self.label2idx[("_").join(cur_obj.split("_")[:-1])]
        point_dict["index"] = index
        point_dict["path"] = obj_dir
        return point_dict

    def __len__(self) -> int:
        return len(self.objs)
    


class ObjaverseFinetune(data.Dataset):
    def __init__(
        self,
        split: str, # train/test/val
    ) -> None:
        with open(f"/data/ziqi/objaverse/labeled/split/{split}.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")
        

    def __getitem__(self, index: int) -> dict:
        name_uid = self.obj_path_list[index]
        file_path = f"/data/ziqi/objaverse/labeled/rendered/{name_uid}/oriented"
        uid = name_uid.split("_")[-1]
        with open(f"{file_path}/masks/merged/mask_labels.txt", "r") as f:
            labels = f.read().splitlines()
        mask_pts = torch.load(f"{file_path}/masks/merged/mask2points.pt").cpu()
        pts_xyz = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/points.pt").cpu()
        normal = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/normals.pt").cpu()
        pts_rgb = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/rgb.pt").cpu()*255

        # subsample to 30 masks per obj
        n_total_masks = mask_pts.shape[0]
        N_MASKS_PER_OBJ = 30
        if n_total_masks >  N_MASKS_PER_OBJ:
            indices = torch.randperm(n_total_masks)[: N_MASKS_PER_OBJ]
            mask_pts = mask_pts[indices,:]
            labels = [labels[i] for i in indices]

        
        point_dict = prep_points_finetune(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy(), mask_pts.numpy())

        ## encode label
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        point_dict['label_embeds'] = text_feat # n_cur_mask, dim_feat, need to be padded
        #self.seed += 1
        return point_dict

    def __len__(self) -> int:
        return len(self.obj_path_list)
    


class ObjaverseFinetuneIoUEval(data.Dataset): # batch size can only be 1 for this!
    def __init__(
        self,
        split: str, # train/test/val
    ) -> None:
        with open(f"/data/ziqi/objaverse/labeled/split/{split}.txt", "r") as f:
            self.obj_path_list = f.read().splitlines()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")

    def __getitem__(self, index: int) -> dict:
        name_uid = self.obj_path_list[index]
        file_path = f"/data/ziqi/objaverse/labeled/rendered/{name_uid}/oriented"
        uid = name_uid.split("_")[-1]
        with open(f"{file_path}/masks/merged/mask_labels.txt", "r") as f:
            labels = f.read().splitlines()

        masks = torch.load(f"{file_path}/masks/merged/allmasks.pt")
        mask_view_idxs = torch.load(f"{file_path}/masks/merged/mask2view.pt")
        pt2face = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/point2face.pt").cpu()
        pix2face = torch.load(f"{file_path}/pix2face.pt").cpu()
        mask_pts = torch.load(f"{file_path}/masks/merged/mask2points.pt").cpu()
        pts_xyz = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/points.pt").cpu()
        normal = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/normals.pt").cpu()
        pts_rgb = torch.load(f"/data/ziqi/objaverse/labeled/points/{uid}/rgb.pt").cpu()*255
        
        point_dict = prep_points_finetune_val(pts_xyz.numpy(), pts_rgb.numpy(), normal.numpy(), mask_pts.numpy(), pt2face.numpy())
        
        ## encode label
        inputs = self.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_feat = self.model.get_text_features(**inputs) # n_masks, feat_dim (768)
        
        #normalize
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)

        point_dict['label_embeds'] = text_feat # n_cur_mask, dim_feat, need to be padded
        point_dict['masks'] = masks
        point_dict['mask_view_idxs'] = mask_view_idxs
        point_dict['pixel2face'] = pix2face
        point_dict['labels'] = labels
        return point_dict

    def __len__(self) -> int:
        return len(self.obj_path_list)