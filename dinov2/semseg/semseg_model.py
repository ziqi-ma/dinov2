# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dinov2.models.mcc.mcc_model import MCCDecoderAttention, MCCDecoderBlock, LayerScale, DecodeXYZPosEmbed
from dinov2.eval.setup import setup_and_build_model
from dinov2.models.pt3.model import offset2batch, batch2offset
#from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath
#from util.pos_embed import get_2d_sincos_pos_embed


def sincos_positional_embedding(positions, num_embeddings):
    """
    Generate sincos positional embeddings for 3D coordinates.

    Args:
        positions (torch.Tensor): A tensor of shape (N, 3) representing the 3D coordinates.
        num_embeddings (int): The number of sine/cosine frequencies to use.

    Returns:
        torch.Tensor: A tensor of shape (N, 6 * num_embeddings) containing the embeddings.
    """
    # Initialize an empty tensor for the embeddings
    N = positions.shape[0]
    embeddings = torch.zeros(N, 6 * num_embeddings, device=positions.device)

    # Create frequency levels
    freqs = 2 ** torch.arange(num_embeddings, dtype=torch.float32, device=positions.device)

    for i in range(3):  # Loop over the 3 dimensions (x, y, z)
        # Compute the sine and cosine embeddings for each frequency
        embeddings[:, 2 * i * num_embeddings: (2 * i + 1) * num_embeddings] = torch.sin(positions[:, i][:, None] * freqs)
        embeddings[:, (2 * i + 1) * num_embeddings: (2 * i + 2) * num_embeddings] = torch.cos(positions[:, i][:, None] * freqs)

    return embeddings


class SemSeg(nn.Module):
    """ Masked Autoencoder with PT3
    """
    def __init__(self, args,
                 out_dim=768, point_embed_dim=32,decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., init_logit_scale=np.log(1 / 0.07), num_pos_embeddings=32, norm_layer=nn.LayerNorm):
        super().__init__()

        self.args = args
        # define logit scale
        self.ln_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder_backbone, _, encoder_embed_dim = setup_and_build_model(args)
        # set gradients false for encoder backbone
        self.freeze_encoder()
        self.cached_enc_feat = None
        self.num_pos_embeddings=num_pos_embeddings

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(
            encoder_embed_dim + num_pos_embeddings*6,
            decoder_embed_dim,
            bias=True
        ) # this is just a projection from encoded features to decoder feature
        self.decoder_pt_embed = nn.Linear(
            point_embed_dim,
            decoder_embed_dim,
            bias=True
        ) # this is just a projection from point embedding from PT3 (32dim) to decoder feature

        self.decoder_feat_embed = nn.Linear(3+3, decoder_embed_dim) # this 3+3 is rgb+normal
        #self.decoder_xyz_embed = DecodeXYZPosEmbed(num_pos_embeddings*6)
        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_query_pts, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            MCCDecoderBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=args.drop_path,
                args=args,
            ) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.fc1 = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.fc2 = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.fc3 = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.fc4 = nn.Linear(decoder_embed_dim, out_dim)
        #self.decoder_pred = nn.Linear(decoder_embed_dim, out_dim, bias=True) # decoder to output dimension

        self.initialize_weights()

    def initialize_weights(self):

        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        #torch.nn.init.normal_(self.cls_token_xyz, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.decoder_embed.apply(self._init_weights)
        self.decoder_pt_embed.apply(self._init_weights)
        self.decoder_blocks.apply(self._init_weights)
        self.decoder_norm.apply(self._init_weights)
        #self.decoder_xyz_embed.apply(self._init_weights)
        self.decoder_feat_embed.apply(self._init_weights)
        self.fc1.apply(self._init_weights)
        self.fc2.apply(self._init_weights)
        self.fc3.apply(self._init_weights)
        self.fc4.apply(self._init_weights)
    
    def decoder_pred_head(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def freeze_encoder(self):
        for param in self.encoder_backbone.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_encoder(self, data_dict):
        # x is a dictionaty with a batch of points, this should go through the PT3 backbone
        encoded_feats_dict = self.encoder_backbone.get_feats(data_dict)
        data_dict["global_feats"] = encoded_feats_dict["x_norm_clstoken"] # this is B*512
        data_dict["patch_feats"] = encoded_feats_dict["x_norm_patchtokens"] # this is total_patches*512, each obj doesn't have the same # of patches
        data_dict["patch_coord"] = encoded_feats_dict["patch_coord"]
        
        # if obj_cum_idx exists, this means we are in test mode where
        # multiple points in one grid will result in multiple copies of the data
        # (i.e. first dict only contains point 0 in every grid, second point 1 in every grid etc.)
        # in this case patch_batch_idx will map to more than 0-63 since all points are appended together
        if "obj_cum_idx" in data_dict:
            # we have 2 options here, one is only keep the first patch features and drop the rest
            # for each object
            keep_all = True
            if not keep_all: # only keep the first
                # need to remove some of patch_feats too
                raise NotImplementedError
            # the other is keep all, and just remap the idxs back to obj idx, i.e. 0-63
            index_mapping = data_dict["obj_cum_idx"] # e.g. [2 5 7...] where how many subobjs are in one obj, cumulated
            subobj_offset = batch2offset(data_dict["patch_batch_idx"]) # this is e.g. [35, 90, ...]
            objlevel_patch_offsets = subobj_offset[index_mapping-1] # this is the obj-level offset e.g. [90, 170, ...], TODO: verify
            objlevel_patch_batch = offset2batch(objlevel_patch_offsets)
            data_dict["patch_batch_idx"] = objlevel_patch_batch
        else:
            data_dict["patch_batch_idx"] = encoded_feats_dict["batch_idx"] # this is 00..011..1...63..63 keeping track which patch is which obj

        # optionally, use point embeddings from backbone
        data_dict["point_embedding"] = self.encoder_backbone.get_embedding(data_dict)["feat"]

        return data_dict

    def forward_decoder(self, data_dict):
        data_dict["rgb_full"] = data_dict["rgb_full"].cuda()
        data_dict["normal_full"] = data_dict["normal_full"].cuda()
        data_dict["xyz_full"] = data_dict["xyz_full"].cuda()


        global_feats = data_dict["global_feats"] # should be B, 512
        patch_feats = data_dict["patch_feats"] # should be n_patches, 512
        patch_batch_idx = data_dict["patch_batch_idx"] # which patch belongs to which object
        patch_coord = data_dict["patch_coord"] # coordinates for each patch

        # no grid sampling, so use full offset for points
        point_batch_idx = offset2batch(data_dict["full_offset"])

        '''
        these are all if we use grid sampling
        # if obj_cum_idx exists, this means we are in test mode where
        # multiple points in one grid will result in multiple copies of the data
        # (i.e. first dict only contains point 0 in every grid, second point 1 in every grid etc.)
        # in this case point_batch_idx will map to more than 0-63 since all points are appended together
        if "obj_cum_idx" in data_dict:
            index_mapping = data_dict["obj_cum_idx"] # e.g. [2 5 7...] where how many subobjs are in one obj, cumulated
            subobj_offset = data_dict["offset"] # this is e.g. [3569, 5000, ...]
            objlevel_point_offsets = subobj_offset[index_mapping-1] # this is the obj-level offset e.g. [5000, 10000, ...], TODO: verify
            point_batch_idx = offset2batch(objlevel_point_offsets)
        else:
            point_batch_idx = offset2batch(data_dict["offset"]) # which point belongs to which object
        '''

        #query_rgbnormal_embedding = data_dict["point_embedding"] # should be n_total_pts, 32
        x = torch.cat([global_feats, patch_feats], dim=0)
        # append positional embedding of the patch coords
        patch_posembedding = sincos_positional_embedding(patch_coord, self.num_pos_embeddings) # n_patches, 32*6=192
        all_pos_embedding = torch.cat([torch.zeros((global_feats.shape[0],patch_posembedding.shape[1])).cuda(), patch_posembedding], dim=0) # n_patched+B, num_pos_embeddings*6
        x = torch.cat([x, all_pos_embedding], dim=1) # this is n_patches+B, 512+num_pos_embeddings*6

        query_rgbnormal = torch.cat([data_dict["rgb_full"], data_dict["normal_full"]], dim=1)
        query_feats_embedding = self.decoder_feat_embed(query_rgbnormal) # n_total_pts, 512
        query_xyz_posembedding = sincos_positional_embedding(data_dict["xyz_full"], self.num_pos_embeddings) # n_total_pts, 32*6=192

        query_embedding = torch.cat([query_feats_embedding, query_xyz_posembedding], dim=1) # n_total_pts, 512+num_pos_embeddings*6
        
        x = torch.cat([x, query_embedding], dim=0)
        # project them both
        x = self.decoder_embed(x) # this should be n_total_patches+B, 512

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, patch_batch_idx, point_batch_idx) #N,decoder_dim(512)

        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred_head(x) # N,768
        # remove cls & seen token
        n_query = point_batch_idx.shape[0]
        pred = pred[-n_query:, :] # n_pts_total, 768
        return pred


    def clear_cache(self):
        self.cached_enc_feat = None

    def forward(self, data_dict):
        data_dict = self.forward_encoder(data_dict)
        pred = self.forward_decoder(data_dict)
        return pred



