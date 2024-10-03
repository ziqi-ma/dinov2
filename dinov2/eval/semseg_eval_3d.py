### NOTE these functions are only used for holdout evaluation (3D iou with 3D ground truth) - don't use them in any way during training/tuning!
import torch
from tqdm import tqdm
import numpy as np
import time
import argparse
from typing import List, Optional
from torch.utils.data import DataLoader
from dinov2.utils.config import setup
import torch
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.data.datasets import collate_fn
from dinov2.data.datasets import ObjaverseEval3D, EvalPartNetE, EvalShapeNetPart
import numpy as np
import random
from dinov2.semseg.semseg_model import SemSeg
from functools import partial
import open3d as o3d


def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=[],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    parser.set_defaults(
        batch_size=64
    )
    return parser

# NOTE: we only care about the ious of actual labeled parts
# the expected label_gt starts from 0 and ends at n_parts
# where 0 is unlabeled (note for Shapenetpart and PartnetE we need to +1!)
# 1,2,...n_parts are actual part labels
# in this function, to avoid cases where there are only very few part names
# and all scores are low, we append a 0 in the end before the softmax
# so essentially the prediction softmax score is of shape (n_pts, n_parts+1)
# where cols 0, ...n_parts-1's correspond to assigning part 1, ... part n_parts
# and col n_part corresponds to the appended score of 0, i.e. it is only
# chosen if all parts have negative score, i.e. this point is unlabeled
# so we add 1 after softmax, making it 1,2,...n_parts, n_parts+1 where n_parts+1 corresponds
# to unlabeled, but when computing IoU we only iterate thru part 1,...n_parts
# so this works out

def compute_3d_iou(pred, # n_pts, feat_dim
                   part_text_embeds, # n_parts, feat_dim
                   temperature,
                   label_gt # n_pts
                   ):
    # the text embedding is normalized to norm 1
    # the pred is not normalized since it's whatever the model outputs
    # we regard anything > 0 as a match and <= as not a match when obtaining masks
    # we can further adjust this if we decide to normalize the pred here (not during training since then the 
    # contrastive loss will be affected)
    # first get each point's logits
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    # append 0 - in case all scores are very low
    logits_append0 = torch.cat([logits, torch.zeros(logits.shape[0],1).cuda()],axis=1)
    pred_softmax = torch.nn.Softmax(dim=1)(logits_append0 * temperature)#[:,:-1]
    pred_cat = pred_softmax.argmax(dim=1) + 1 # here, 1,...n_part correspond to actual part assignment, n_part+1 corresponds to unlabeled

    #label_gt = label_gt + 1
    acc = ((pred_cat == label_gt)*1).sum() / pred_cat.shape[0]

    pred_np = pred_cat.cpu().numpy()
    label_np = label_gt.cpu().numpy()

    # get per part iou
    part_ious = []
    for part in range(part_text_embeds.shape[0]):
        I = np.sum(np.logical_and(pred_np == part+1, label_np == part+1))
        U = np.sum(np.logical_or(pred_np == part+1, label_np == part+1))
        if U == 0:
            pass
        else:
            iou = I / float(U)
            part_ious.append(iou)
    mean_iou = np.mean(part_ious)
    return acc.item(), mean_iou


# we upsample almost always because grid sampling will only keep 1 point per grid
def compute_3d_iou_upsample(
        pred, # n_subsampled_pts, feat_dim
        part_text_embeds, # n_parts, feat_dim
        temperature,
        gt_subsample, # n_subsample_pts,
        xyz_sub,
        xyz_full, # n_pts, 3
        gt_full, # n_pts,
        DISTANCE_CUTOFF=1, # this is how many neighbors you want, i.e. all neighbors within DISTANCE_CUTOFF of closest pt
        N_CHUNKS=1 # this is how many chunks so that subset fits in GPU, 3 for the biggest partnete
        ):
    # the text embedding is normalized to norm 1
    # the pred is not normalized since it's whatever the model outputs
    # we regard anything > 0 as a match and <= as not a match when obtaining masks
    # we can further adjust this if we decide to normalize the pred here (not during training since then the 
    # contrastive loss will be affected)
    # first get each point's logits
    logits = pred @ part_text_embeds.T # n_pts, n_mask

    # append 0 - in case all scores are very low
    logits_append0 = torch.cat([logits, torch.zeros(logits.shape[0],1).cuda()],axis=1)
    pred_softmax = torch.nn.Softmax(dim=1)(logits_append0 * temperature)

    # get the acc and iou normally on the subsample first, can remove this later
    sub_miou = 0
    sub_macc = 0
    if True:
        pred_sub = pred_softmax.argmax(dim=1) + 1 # here, 1,...n_part correspond to actual part assignment, n_part+1 corresponds to unlabeled
        #label_gt = label_gt + 1
        acc = ((pred_sub == gt_subsample)*1).sum() / pred_sub.shape[0]
        pred_np = pred_sub.cpu().numpy()
        label_np = gt_subsample.cpu().numpy()

        # get per part iou
        part_ious = []
        for part in range(part_text_embeds.shape[0]):
            I = np.sum(np.logical_and(pred_np == part+1, label_np == part+1))
            U = np.sum(np.logical_or(pred_np == part+1, label_np == part+1))
            if U == 0:
                pass
            else:
                iou = I / float(U)
                part_ious.append(iou)
        sub_miou = np.mean(part_ious)
        sub_macc = acc.item()

    # method 1 knn, this is too slow, 212s/obj
    # this is for each pt in the big pt cloud, find knn of labeled small pt cloud pts
    # the slow way: n_full log(n_sub)
    '''
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(xyz_full.numpy())
    pcd_sub = o3d.geometry.PointCloud()
    pcd_sub.points = o3d.utility.Vector3dVector(xyz_sub.cpu().numpy())
    # get nearest neighbors per point
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
    all_probs = torch.zeros((xyz_full.shape[0], part_text_embeds.shape[0]+1)).cuda()
    N_NEIGHBORS=200
    for i in range(xyz_full.shape[0]):
        [k, idxs, dissqr] = pcd_tree.search_knn_vector_3d(pcd_full.points[i], N_NEIGHBORS)
        # weigh by inverse to distance
        idxs = np.asarray(idxs)
        dist = np.sqrt(np.asarray(dissqr))
        for j in range(k):
            all_probs[i,:] += pred_softmax[idxs[j],:]/dist[j]
    '''

    # method 2 faster to do if distance=1 (i.e. knn with k=1)
    if DISTANCE_CUTOFF == 1:
        chunk_len = xyz_full.shape[0]//N_CHUNKS+1
        closest_idx_list = []
        for i in range(N_CHUNKS):
            cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
            dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
            cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
            min_idxs = torch.min(cur_dist, 1)[1]
            del cur_dist
            closest_idx_list.append(min_idxs)
        all_nn_idxs = torch.cat(closest_idx_list,axis=0)
        # just inversely weight all points
        all_probs = pred_softmax[all_nn_idxs]
    else:
        # method 2 inverse weigh all pts
        # chunk this, in case of OOM
        chunk_len = xyz_full.shape[0]//N_CHUNKS+1
        dist_list = []
        for i in range(N_CHUNKS):
            cur_chunk = xyz_full[chunk_len*i:chunk_len*(i+1)]
            dist_all = (xyz_sub.unsqueeze(0) - cur_chunk.cuda().unsqueeze(1))**2 # 300k,5k,3
            cur_dist = (dist_all.sum(dim=-1))**0.5 # 300k,5k
            dist_list.append(cur_dist)
        full_sub_dist = torch.cat(dist_list,axis=0)
        # just inversely weight all points
        weights = 1/full_sub_dist
        del full_sub_dist
        # kill very small weights
        maxweight = weights.max(dim=1)[0].view(-1,1) # take values
        weights[weights<maxweight/DISTANCE_CUTOFF] = 0 # ignore the ones that are over twice as far as the closest point
        all_probs = weights @ pred_softmax
    

    
    # now argmax
    pred_full = all_probs.argmax(dim=1).cpu() + 1 # here, 1,...n_part correspond to actual part assignment, n_part+1 corresponds to unlabeled
    #label_gt = label_gt + 1
    acc = ((pred_full == gt_full)*1).sum() / pred_full.shape[0]
    pred_np = pred_full.numpy()
    label_np = gt_full.numpy()

    # get full iou
    part_ious = []
    for part in range(part_text_embeds.shape[0]):
        I = np.sum(np.logical_and(pred_np == part+1, label_np == part+1))
        U = np.sum(np.logical_or(pred_np == part+1, label_np == part+1))
        if U == 0:
            pass
        else:
            iou = I / float(U)
            part_ious.append(iou)
    full_miou = np.mean(part_ious)
    full_macc = acc.item()
    return full_miou, full_macc, sub_miou, sub_macc

'''
def evaluate3d(model, dataloader, set_name): # evaluate loader can only have batch size=1
    temperature = np.exp(model.ln_logit_scale.detach().cpu())
    iou_list = []
    acc_list = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {set_name}-set"):
            print(data["file_path"])
            for key in data.keys():
                if "full" in key:
                    continue
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            net_out = model(data)
            text_embeds = data['label_embeds']
            gt = data["gt"]
            acc, iou = compute_3d_iou_upsample(net_out, # n_pts, feat_dim
                                      text_embeds, # n_parts, feat_dim
                                      temperature,
                                      gt # n_pts
                                      )
            iou_list += [iou]
            print(iou)
            acc_list += [acc]
    miou = np.mean(iou_list)
    macc = np.mean(acc_list)
    return miou, macc
'''

def evaluate3d(model, dataloader, category, DISTANCE_CUTOFF=1, N_CHUNKS=1): # evaluate loader can only have batch size=1
    temperature = np.exp(model.ln_logit_scale.detach().cpu())
    iou_list = []
    acc_list = []
    sub_iou_list = []
    sub_acc_list = []
    with torch.no_grad():
        for data in dataloader:
            for key in data.keys():
                if "full" in key and "gt" not in key:
                    continue
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            net_out = model(data)
            xyz_sub = data["coord"]
            text_embeds = data['label_embeds']
            gt_subsample = data["gt"]
            gt_full = data["gt_full"]
            xyz_full = data["xyz_full"]
            acc, iou = compute_3d_iou(
                net_out, # n_subsampled_pts, feat_dim
                text_embeds, # n_parts, feat_dim
                temperature,
                gt_full) 
            '''
            iou, acc, iou_sub, acc_sub = compute_3d_iou_upsample(
                net_out, # n_subsampled_pts, feat_dim
                text_embeds, # n_parts, feat_dim
                temperature,
                gt_subsample, # n_subsample_pts,
                xyz_sub,
                xyz_full, # n_pts, 3
                gt_full,#n_pts
                DISTANCE_CUTOFF=DISTANCE_CUTOFF,
                N_CHUNKS=N_CHUNKS) 
            '''
            iou_list += [iou]
            acc_list += [acc]
    miou = np.mean(iou_list)
    macc = np.mean(acc_list)
    return miou, macc#, subiou, subacc


def eval_category_partnete(model, category, subset, apply_rotation, DISTANCE_CUTOFF=1):
    test_data = EvalPartNetE(category, subset=subset, apply_rotation=apply_rotation)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    imiou, imacc, imiousub, imaccsub = evaluate3d(model, test_loader, category, DISTANCE_CUTOFF=DISTANCE_CUTOFF, N_CHUNKS=3)
    etime = time.time()
    return imiou, imacc, imiousub, imaccsub, etime-stime

def eval_partnete_all(subset, apply_rotation, DISTANCE_CUTOFF=1):
    mious = []
    submious = []
    alltime = 0
    for category in partnete_categories.keys():
        miou, _, submiou, _, cat_time = eval_category_partnete(model, category, subset, apply_rotation, DISTANCE_CUTOFF=DISTANCE_CUTOFF)
        print(f"category {category} miou: {miou}, sub miou: {submiou}, time {cat_time}")
        mious.append(miou)
        submious.append(submiou)
        alltime += cat_time
    mean_iou = np.mean(mious)
    mean_submious = np.mean(submious)
    print(f"overall miou {mean_iou}, subsampled {mean_submious}, time {alltime}")
    return


def eval_category_shapenetpart(model, category, subset, apply_rotation, DISTANCE_CUTOFF=1):
    test_data = EvalShapeNetPart(category, subset=subset, apply_rotation=apply_rotation)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=0, 
                             drop_last=False)
    stime = time.time()
    imiou, imacc, imiousub, imaccsub = evaluate3d(model, test_loader, category, DISTANCE_CUTOFF=DISTANCE_CUTOFF, N_CHUNKS=1)
    etime = time.time()
    return imiou, imacc, imiousub, imaccsub, etime-stime


def eval_shapenetpart_all(subset, apply_rotation, DISTANCE_CUTOFF=1):
    mious = []
    submious = []
    time_all = 0
    for category in shapenetpart_categories.keys():
        miou, _, submiou, _, cat_time = eval_category_shapenetpart(model, category, subset, apply_rotation, DISTANCE_CUTOFF=DISTANCE_CUTOFF)
        print(f"category {category} miou: {miou}, sub miou: {submiou}, time {cat_time}")
        mious.append(miou)
        submious.append(submiou)
        time_all += cat_time
    mean_iou = np.mean(mious)
    mean_submious = np.mean(submious)
    print(f"overall miou {mean_iou}, gridsampled {mean_submious}, time {time_all}")
    return


def eval_3d_objaverse(model, split, DISTANCE_CUTOFF=1):
    test_data = ObjaverseEval3D(split=split)
    test_loader = DataLoader(test_data, 
                             batch_size=1, 
                             shuffle=False,
                             collate_fn=collate_fn, 
                             num_workers=5, 
                             drop_last=False)
    stime = time.time()
    imiou, imacc = evaluate3d(model, test_loader, split, DISTANCE_CUTOFF=DISTANCE_CUTOFF, N_CHUNKS=1)
    etime = time.time()
    print(f"category {split} miou: {imiou}, time {etime-stime}")#, sub miou: {imiousub}, time {etime-stime}")

       
def load_model(checkpt_path):
    model = SemSeg(
        args,
        out_dim=768,
        point_embed_dim=32,
        decoder_embed_dim=512, init_logit_scale=np.log(1 / 0.07), decoder_depth=4, decoder_num_heads=2,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    ).cuda()
    state_dict = torch.load(checkpt_path)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model

if __name__ == '__main__':
    
    partnete_categories = {'Bottle':16, 'Box':17, 'Bucket':18, 'Camera':19, 'Cart':20, 'Chair':21, 'Clock':22,
            "CoffeeMachine": 23, 'Dishwasher': 24, 'Dispenser': 25, "Display": 26, 'Eyeglasses': 27,
            'Faucet': 28, "FoldingChair": 29, "Globe": 30, "Kettle":31, "Keyboard": 32, "KitchenPot": 33,
            "Knife": 34, "Lamp": 35, "Laptop": 36, "Lighter": 37, "Microwave": 38, "Mouse": 39, "Oven": 40,
            "Pen": 41, "Phone": 42, "Pliers": 43, "Printer": 44, "Refrigerator": 45, "Remote": 46,
            "Safe": 47, "Scissors": 48, "Stapler": 49, "StorageFurniture": 50, "Suitcase": 51,
            "Switch": 52, "Table": 53, "Toaster": 54, "Toilet": 55, "TrashCan": 56, "USB": 57,
            "WashingMachine": 58, "Window": 59, "Door": 60}
    
    
    shapenetpart_categories = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
     'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
     'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
    

    description = "holdout eval of decoder+encoder for SemSeg3D"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    args.pretrained_weights = "/data/ziqi/training_checkpts/1e6new/eval/training_1199/teacher_checkpoint.pth"# this is at least much more rotational invariant #"/data/ziqi/training_checkpts/debugall/eval/training_1519/teacher_checkpoint.pth"
    args.checkpoint_path="/data/ziqi/training_checkpts/dinofinetune4/checkpoint5.pt"
    args.drop_path=0
    args.seed = 123
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint_path)
    
    eval_3d_objaverse(model, "unseen", DISTANCE_CUTOFF=1) # use label of nearest neighbor
    #eval_partnete_all(subset=True, apply_rotation=True, DISTANCE_CUTOFF=1)
    #eval_shapenetpart_all(subset=False, apply_rotation=True, DISTANCE_CUTOFF=1)
    