# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection
import torch.nn.functional as F
from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from dinov2.data.datasets import collate_fn
import open3d as o3d

logger = logging.getLogger("dinov2")


def visualize_pts(pts, labels): # pts is n*3, colors is n, 0 - n-1 where 0 is unlabeled
    part_num = labels.max()
    cmap_matrix = torch.tensor([[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5],[0,0.5,0.5],
                [0.1,0.2,0.3],[0.2,0.5,0.3], [0.6,0.3,0.2], [0.5,0.3,0.5],
                [0.6,0.7,0.2],[0.5,0.8,0.3]]).cuda()[:part_num+1,:]
    colors = ["white", "red", "green", "blue", "yellow", "magenta", "cyan","grey", "olive",
                "purple", "teal", "navy", "darkgreen", "brown", "pinkpurple", "yellowgreen", "limegreen"]
    caption_list=[f"{i}:{colors[i]}" for i in range(part_num+1)]
    onehot = F.one_hot(labels.long(), num_classes=part_num+1) * 1.0 # n_pts, part_num+1, each row 00.010.0, first place is unlabeled (0 originally)
    pts_rgb = torch.matmul(onehot, cmap_matrix) # n_pts,3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pts_rgb.cpu().numpy())
    o3d.visualization.draw_plotly([pcd])
    print(caption_list)

class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for data in metric_logger.log_every(data_loader, 10):
        targets = data["label"].cuda()
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)
        outputs = model(data)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dinohead, dataset, batch_size, num_workers, gather_on_cpu=False):
    #dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset)
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.EPOCH,
        sampler_size = sample_count,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    return extract_features_with_dataloader(model, dinohead, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, dinohead, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for data in metric_logger.log_every(data_loader, 10):
        labels_rank = data["label"].cuda()
        index = data["index"].cuda()
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)
        features_rank = model(data).float()
        #features_rank = dinohead(features_rank).float()
        #features_rank = F.softmax((features_rank/0.07), dim=-1)
        #features_rank = nn.functional.normalize(features_rank, dim=1, p=2)

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
