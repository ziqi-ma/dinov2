# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from .transform import *
import torch.utils.data as data
import os
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
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='z',center=[0, 0, 0],p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='y',p=1)(data_dict)
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.15)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.15)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    #data_dict = SphereCrop(sample_rate=0.6, mode='random')(data_dict)
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
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='z',center=[0, 0, 0],p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='y',p=1)(data_dict)
    data_dict = RandomScale(scale=[0.9, 1.1])(data_dict)
    data_dict = RandomFlip(p=0.5)(data_dict)
    data_dict = RandomJitter(sigma=0.005, clip=0.02)(data_dict)
    data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
    data_dict = ChromaticTranslation(p=0.95, ratio=0.15)(data_dict)
    data_dict = ChromaticJitter(p=0.95, std=0.15)(data_dict)
    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict)
    #data_dict = SphereCrop(sample_rate=0.2, mode='random')(data_dict)
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
    
    # try rotate
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='z',center=[0, 0, 0],p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='x',p=1)(data_dict)
    data_dict = RandomRotate(angle=[-3.14, 3.14],axis='y',p=1)(data_dict)

    data_dict = GridSample(grid_size=0.02,hash_type='fnv',mode='train',return_grid_coord=True)(data_dict) # mode train is used in original code, text will subsample points n times and create many samples out of one sample
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    #data_dict = Add(keys_dict=dict(condition='S3DIS'))(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(keys=('coord', 'grid_coord'),
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
    for query_idx in idx_list:
        idx_start = idx_offsets[query_idx]
        idx_end = idx_offsets[query_idx+1]
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
        local_flattened_batch = []
        for item in batch:
            local_flattened_batch += item["local_augs"]
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
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


class ObjaverseAugmented(data.Dataset):
    def __init__(
        self,
        *,
        n_local_crops:int,
        split: str, # train/test/val
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
        split: str, # train/test/val
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
        self.objs = [os.listdir(self.dirpath)[idx] for idx in test_subset_idxs]
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