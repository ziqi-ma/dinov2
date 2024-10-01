import argparse
from typing import List, Optional
from dinov2.semseg.semseg_model import SemSeg
from functools import partial
import torch
import torch.optim as optim
import torch.nn as nn
from dinov2.data.datasets.objaverse import ObjaverseFinetune, ObjaverseFinetuneIoUEval, collate_fn
from torch.utils.data import DataLoader
from dinov2.loss.pointcontrastive import PointContrastive
from dinov2.eval.semseg_eval import compute_overall_iou_objwise, viz_pred_mask
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import wandb
from dinov2.utils.utils import CosineScheduler

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


def evaluate_miou(model, objwise_loader, n_epoch, set_name, temperature, visualize_interval=1, visualize_idxs=[20,25,55,80,139]):
    iou_list = []
    j = 0
    with torch.no_grad():
        for data in tqdm(objwise_loader, desc=f"Evaluating {set_name}-set"):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)

            net_out = model(data)
            text_embeds = data['label_embeds']
            masks = data['masks']
            mask_view_idxs = data["mask_view_idxs"]
            point2face = data['point2face']
            pix2face = data['pixel2face']
            labels = data['labels']
            
            m = AutoModel.from_pretrained("google/siglip-base-patch16-224") # dim 768 #"google/siglip-so400m-patch14-384")
            tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")#"google/siglip-so400m-patch14-384")
            inputs = tokenizer(labels[0], padding="max_length", return_tensors="pt")
            with torch.no_grad():
                text_feat = m.get_text_features(**inputs) # n_masks, feat_dim (768)
        
            #normalize
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-12)
            # we index 0 into everything because batch size = 1 for evaluation
            iou = compute_overall_iou_objwise(pred=net_out, # n_pts, feat_dim
                                              text_embeds = text_embeds, # n_masks, feat_dim
                                              masks=masks, #n_masks, h, w - binary in 2d
                                              mask_view_idxs = mask_view_idxs, # n_masks each has a view index, -1 for padding
                                              # metadata below
                                              point2face = point2face, # n_pts
                                              pixel2face = pix2face, # 10,H,W
                                              temperature= temperature
                                              )
            iou_list += [iou]

            # visualize if fall on visualization index

            if n_epoch % visualize_interval == 0 and j in visualize_idxs:
                viz_pred_mask(pred=net_out,
                            text_embeds = text_embeds, # n_masks, feat_dim
                            texts = [[x] for x in labels[0]],
                            masks=masks, #n_masks, h, w - binary in 2d
                            mask_view_idxs = mask_view_idxs, # n_masks each has a view index, -1 for padding
                            point2face = point2face, # n_pts
                            pixel2face = pix2face,# 10,H,W
                            n_epoch = n_epoch, # which epoch we are evaluating
                            obj_visualize_idx = j, # which object we are evaluating
                            prefix = f"mccfinetune-{set_name}",
                            temperature=temperature
                            )
            j += 1
    miou = np.mean(iou_list)
    return miou


def evaluate_loss(model, dataloader, criterion):
    loss_list = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda(non_blocking=True)
            mask_points = data['mask2pt'] 
            mask_embeds = data['label_embeds']
            pt_offset = data['full_offset'] # no grid subsampling
            net_out = model(data) #net_out=[total_pts_batch, dim_feat]    
            loss = criterion(net_out,
                             pt_offset, # offset is tensor of shape B+1, marking starting idx of each obj
                             mask_embeds,
                             mask_points,
                             model.ln_logit_scale)
            loss_list += [loss.item()]
    mloss = np.mean(loss_list)
    return mloss


def train_semseg_model(args):
    model = SemSeg(
        args,
        out_dim=768,
        point_embed_dim=32,
        decoder_embed_dim=512, init_logit_scale=np.log(1 / 0.07), decoder_depth=4, decoder_num_heads=2,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).cuda()
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    run = wandb.init(project="finetune-mcc", config=args)
    train_data = ObjaverseFinetune("train")
    val_data = ObjaverseFinetune("val")
    objwise_train_data = ObjaverseFinetuneIoUEval("train")
    objwise_val_data = ObjaverseFinetuneIoUEval("val")
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              num_workers=0, # this needs to be 0 since we do text embedding in loader which requires cuda, cuda cannot have multiple workers
                              drop_last=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False, 
                            collate_fn=collate_fn,
                            num_workers=0, # this needs to be 0 since we do text embedding in loader which requires cuda, cuda cannot have multiple workers
                            drop_last=False)
    train_iou_loader = DataLoader(objwise_train_data, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_iou_loader = DataLoader(objwise_val_data, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    iter_per_epoch = len(train_data) // args.batch_size
    lr = dict(
        base_value=args.lr,
        final_value=args.lr/4,
        total_iters=args.n_epoch*iter_per_epoch,
        warmup_iters=args.n_epoch*iter_per_epoch // 10,
        start_warmup_value=1e-6,
    )
    lr_schedule = CosineScheduler(**lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.n_epoch, eta_min=args.lr/100)
    criterion = PointContrastive()
    iter = 0
    

    for epoch in range(args.n_epoch):
        epoch = epoch + 1
        loss_epoch_current = []
        
        for data in (tqdm(train_loader, desc=f"Training epoch: {epoch}/{args.n_epoch}")):
            lr = lr_schedule[iter]
            for param_group in opt.param_groups:
                param_group["lr"] = lr

            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and "full" not in key:
                    data[key] = data[key].cuda(non_blocking=True)

            mask_points = data['mask2pt'] 
            mask_embeds = data['label_embeds']
            pt_offset = data['full_offset'] # no grid subsampling, use full offset
        
            net_out = model(data) #net_out=[total_pts_batch, dim_feat]
            
            pt_offset = pt_offset.cuda()
            loss = criterion(net_out,
                             pt_offset, # offset is tensor of shape B+1, marking starting idx of each obj
                             mask_embeds,
                             mask_points,
                             model.ln_logit_scale)
            
            loss_epoch_current.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            iter += 1
            epoch_loss_avg = np.around(np.mean(loss_epoch_current), decimals=4)
            log_dict = {"train loss":epoch_loss_avg, "temperature":np.exp(model.ln_logit_scale.item()), "learning rate": lr}
            
            
            if iter % iter_per_epoch == 0:
                # evaluate miou
                miou_val = evaluate_miou(model, val_iou_loader, epoch, "val", temperature = torch.exp(model.ln_logit_scale.detach()))
                mloss_val = evaluate_loss(model, val_loader, criterion)
                
                # this takes 10 hours to eval all of train set, too long
                #miou_train = evaluate_miou(model, train_iou_loader, epoch, "train", temperature = torch.exp(model.ln_logit_scale))
                
                print(f"val: loss {mloss_val}, miou {miou_val}; train: loss {epoch_loss_avg}")
                #log_dict["train miou"]=miou_train
                log_dict["val loss"]=mloss_val
                log_dict["val miou"]=miou_val
            
            
            if iter % 20 == 0 or iter % iter_per_epoch == 0:
                wandb.log(log_dict, step=iter, commit=True)

            
            if iter % (iter_per_epoch // 3) == 0:
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':opt.state_dict(),
                            'loss':loss},
                            f"{args.output_dir}/checkpoint{epoch}.pt")
            


if __name__ == "__main__":
    description = "train segmentation model with PT3 encoder and MCC decoder"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    args.pretrained_weights = "/data/ziqi/training_checkpts/1e6new/eval/training_1199/teacher_checkpoint.pth"# this is at least much more rotational invariant #"/data/ziqi/training_checkpts/debugall/eval/training_1519/teacher_checkpoint.pth"
    args.drop_path=0
    args.lr=2e-4
    args.n_epoch=10
    args.batch_size=6
    args.seed = 123
    torch.manual_seed(args.seed)
    train_semseg_model(args)