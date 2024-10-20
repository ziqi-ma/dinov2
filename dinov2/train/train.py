# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
import numpy as np
from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
import wandb
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.eval.knn import eval_knn_with_model, fetch_eval_k_neighbors
from dinov2.data.datasets import ObjaverseAugmented, ObjaverseEval, ObjaverseEvalSubset, collate_fn, visualize_data
from dinov2.train.ssl_meta_arch import SSLMetaArch
import random

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")

def compute_total_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_false",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.") 

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg, lr):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=lr,#cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=int(cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH),
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : int(cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH)
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, test_loader, teacher_temp, train_knn_dataset_rand,
            val_knn_dataset_all, val_knn_dataset_picked, val_knn_dataset_rand,
            train_knn_dataset_rand_rot, val_knn_dataset_all_rot, val_knn_dataset_picked_rot,
            val_knn_dataset_rand_rot, iteration):
    
    new_state_dict = model.teacher.state_dict()
    iterstring = str(iteration)
    eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)

    if distributed.is_main_process():
        os.makedirs(eval_dir, exist_ok=True)
        # save checkpoint
        model_statedict_path = os.path.join(eval_dir, "model_checkpoint.pth")
        print(f"saving model at {model_statedict_path}")
        torch.save(model.state_dict(), model_statedict_path)

    # TODO: figure out parallelization stuff, this is in the parallelized context
    global_loss_list = []
    local_loss_list = []
    with torch.no_grad():
        for data in test_loader:
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, backward=False)
            global_loss_list.append(loss_dict["dino_global_crops_loss"].item())
            local_loss_list.append(loss_dict["dino_local_crops_loss"].item())
    mean_global_loss = np.mean(global_loss_list)
    mean_local_loss = np.mean(local_loss_list)
    
    with torch.no_grad():
        chamfer_picked, acc_picked = fetch_eval_k_neighbors(
            model.teacher.backbone,
            None,
            val_knn_dataset_all,
            val_knn_dataset_picked,
            10,
            64,
            5, # num workers
            gather_on_cpu=args.gather_on_cpu,
            printout=True
        )
        chamfer_picked_rot, acc_picked_rot = fetch_eval_k_neighbors(
            model.teacher.backbone,
            None,
            val_knn_dataset_all_rot,
            val_knn_dataset_picked_rot, # expect test dataset to be small
            10,
            64,
            5,
            gather_on_cpu=args.gather_on_cpu,
            printout=True
        )

        chamfer_rand, acc_rand = fetch_eval_k_neighbors(
            model.teacher.backbone,
            None,
            train_knn_dataset_rand,
            val_knn_dataset_rand, # expect test dataset to be small
            10,
            64,
            5,
            gather_on_cpu=args.gather_on_cpu,
            printout=False
        )

        chamfer_rand_rot, acc_rand_rot = fetch_eval_k_neighbors(
            model.teacher.backbone,
            None,
            train_knn_dataset_rand_rot,
            val_knn_dataset_rand_rot, # expect test dataset to be small
            10,
            64,
            5,
            gather_on_cpu=args.gather_on_cpu,
            printout=False
        )
    knn_res = {
        "chamfer_picked":chamfer_picked,
        "acc_picked":acc_picked,
        "chamfer_picked_rot":chamfer_picked_rot,
        "acc_picked_rot":acc_picked_rot,
        "chamfer_rand":chamfer_rand,
        "acc_rand":acc_rand,
        "chamfer_rand_rot":chamfer_rand_rot,
        "acc_rand_rot":acc_rand_rot
    }

    return mean_global_loss, mean_local_loss, knn_res




def do_train(cfg, lr, model, resume=False, resume_path=None, pickup_iter=None):

    start_iter = 1
    if resume and resume_path:
        model.load_state_dict(torch.load(resume_path))
        print(f"loaded model from {resume_path}")
        if pickup_iter:
            start_iter = pickup_iter

    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg, lr)

    # checkpointer
    #checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    #start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    #print(start_iter)
    

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    #periodic_checkpointer = PeriodicCheckpointer(
        #checkpointer,
        #period=3 * OFFICIAL_EPOCH_LENGTH,
        #max_iter=max_iter,
        #max_to_keep=3,
    #)

    # setup data loader
    train_dataset = ObjaverseAugmented(n_local_crops=cfg.crops.local_crops_number, split='train', root="/data/ziqi/objaverse/pretrain")
    val_dataset = ObjaverseAugmented(n_local_crops=cfg.crops.local_crops_number, split='test', root="/data/ziqi/objaverse/pretrain")
    
    # these are for comprehensive evaluation
    train_knn_dataset_rand = ObjaverseEvalSubset(split='train', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=range(5000), rotate=False)
    val_knn_dataset_all = ObjaverseEval(split='test', root="/data/ziqi/objaverse/pretrain", rotate=False)
    val_knn_dataset_picked = ObjaverseEvalSubset(split='test', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=[], rotate=False)
    val_knn_dataset_rand = ObjaverseEvalSubset(split='test', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=range(100), rotate=False) # the ordering is random so first 100 suffices
    train_knn_dataset_rand_rot = ObjaverseEvalSubset(split='train', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=range(5000), rotate=True)
    val_knn_dataset_all_rot = ObjaverseEval(split='test', root="/data/ziqi/objaverse/pretrain", rotate=True)
    val_knn_dataset_picked_rot = ObjaverseEvalSubset(split='test', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=[], rotate=True)
    val_knn_dataset_rand_rot = ObjaverseEvalSubset(split='test', root="/data/ziqi/objaverse/pretrain", test_subset_idxs=range(100), rotate=True)
    # sampler_type = SamplerType.INFINITE
    train_sampler_type = SamplerType.EPOCH
    val_sampler_type = SamplerType.EPOCH
    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=train_sampler_type,
        sampler_size = len(train_dataset),
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(len(train_dataset))
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=val_sampler_type,
        sampler_size = len(val_dataset),
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    for epoch in range(cfg.optim["epochs"]):
        print(f"epoch {epoch}")
        for data in train_loader:
            current_batch_size = data["offset"].shape[0] / 2
            if iteration > max_iter:
                return

            # apply schedules

            lr = lr_schedule[iteration]
            wd = wd_schedule[iteration]
            mom = momentum_schedule[iteration]
            teacher_temp = teacher_temp_schedule[iteration]
            last_layer_lr = last_layer_lr_schedule[iteration]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            # compute losses

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

            # clip gradients

            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()

            
            # perform teacher EMA update
            model.update_teacher(mom)
            
            # logging

            if distributed.get_global_size() > 1:
                for v in loss_dict.values():
                    torch.distributed.all_reduce(v)
            loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

            if math.isnan(sum(loss_dict_reduced.values())):
                logger.info("NaN detected")
                raise AssertionError
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            total_grad_norm = compute_total_grad_norm(model)

            metric_logger.update(lr=lr)
            metric_logger.update(wd=wd)
            metric_logger.update(mom=mom)
            metric_logger.update(last_layer_lr=last_layer_lr)
            metric_logger.update(current_batch_size=current_batch_size)
            metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

            # log more for debugging
            loss_dict_reduced["lr"] = lr
            loss_dict_reduced["wd"] = wd
            loss_dict_reduced["mom"] = mom
            loss_dict_reduced["teacher temp"]=teacher_temp
            loss_dict_reduced["last layer lr"]=last_layer_lr
            loss_dict_reduced["total grad norm"]=total_grad_norm
            

            # DEBUG LOSS SPIKES
            '''
            if losses_reduced > prev_loss * 2:
                # spikes!
                print(f"spiking at iteration {iteration}")
                print(data["path"])
                # save data
                net = torch.nn.Linear(2, 2)
                d = net.state_dict()
                d.update(data)
                torch.save(d, f"spikedata5-{iteration}.pth")
            prev_loss = losses_reduced
            '''

            
            # checkpointing and testing
            
            if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
                #train_global_loss, train_local_loss, train_knn_top1, train_knn_top5 = do_test(cfg, model, val_loader, teacher_temp, train_knn_dataset, train_knn_dataset, f"training_{iteration}")
                val_global_loss, val_local_loss, knn_res = do_test(cfg,
                                                                   model,
                                                                   val_loader,
                                                                   teacher_temp,
                                                                   train_knn_dataset_rand,
                                                                   val_knn_dataset_all,
                                                                   val_knn_dataset_picked,
                                                                   val_knn_dataset_rand,
                                                                   train_knn_dataset_rand_rot,
                                                                   val_knn_dataset_all_rot,
                                                                   val_knn_dataset_picked_rot,
                                                                   val_knn_dataset_rand_rot,
                                                                   f"training_{iteration}")
                metric_logger.update(val_global_loss=val_global_loss)
                metric_logger.update(val_local_loss=val_local_loss)
                torch.cuda.synchronize()
                print(knn_res)
                loss_dict_reduced["val global loss"] = val_global_loss
                loss_dict_reduced["val local loss"] = val_local_loss
                # add to wandb logging
                for key in knn_res.keys():
                    loss_dict_reduced[key] = knn_res[key]
                
            wandb.log(loss_dict_reduced, step=iteration, commit=True)
            
            #periodic_checkpointer.step(iteration)

            iteration = iteration + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    run = wandb.init(project="backbone_pretrain", config=args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    if args.no_resume:
        do_train(cfg, args.lr, model, resume=not args.no_resume)
    else:
        do_train(cfg, args.lr, model, resume=not args.no_resume, resume_path=args.resume_path, pickup_iter=args.pickup_iter)



if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    seed=123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    args.gather_on_gpu = False
    args.lr = 2e-5 # this supercedes config since there is some calculation if you do config
    args.no_resume=True
    #args.resume_path="/data/ziqi/training_checkpts/pretrain_encdec3/eval/training_199/model_checkpoint.pth"
    #args.pickup_iter=199
    main(args)
