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

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath

#from util.pos_embed import get_2d_sincos_pos_embed

class MCCDecoderAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.args = args
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, patch_idx, point_idx):
        N, C = x.shape # C is 1024 i.e. embedding dimension, N is total number of points+localpatches+globalfeatures in the batch
        query_size = point_idx.shape[0]
        # TODO: change reshape!
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3) # 3,nh,N,dim', dim'=dim/nh
        q, k, v = qkv.unbind(0) # each nh,N, dim'
        attn = (q @ k.transpose(-2, -1)) * self.scale # nh,N,N

        mask = torch.zeros((1, N, N), device=attn.device)
        # first mask out the point2point part only keep identity
        mask[:, :, -query_size:] = float('-inf')
        # get identity of query pts
        for i in range(query_size):
            mask[:, -(i + 1), -(i + 1)] = 0
        
        # then mask out all the feature2point
        mask[:,-query_size:,:-query_size] = float('-inf')
        # get the global masks, each point corresponding to point_idx
        mask[:,torch.arange(query_size)+N-query_size, point_idx] = 0
        # then get the local masks, each point corresponding to all patches of its idx
        n_patches = patch_idx.max()+1
        for obj in range(n_patches):
            pts = ((point_idx==obj)*1).nonzero() # point indices
            patch = ((patch_idx==obj)*1).nonzero() # patch indices
            patch_min = patch.min()+n_patches
            patch_max = patch.max()+n_patches
            mask[:,pts+N-query_size,patch_min:patch_max+1] = 0

        attn = attn + mask
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0,1).reshape(N, C) # N,nh,dim'->N,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MCCDecoderBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = MCCDecoderAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, args=args)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, patch_idx, point_idx):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), patch_idx, point_idx)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

'''
class XYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.win_size = 8

        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, seen_xyz, valid_seen_xyz):
        emb = self.pos_embed(seen_xyz)

        emb[~valid_seen_xyz] = 0.0
        emb[~valid_seen_xyz] += self.invalid_xyz_token

        B, H, W, C = emb.shape
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)

        emb = emb + self.two_d_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)
'''

class DecodeXYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Linear(3, embed_dim)

    def forward(self, unseen_xyz):
        return self.pos_embed(unseen_xyz)


def shrink_points_beyond_threshold(xyz, threshold):
    xyz = xyz.clone().detach()
    dist = (xyz ** 2.0).sum(axis=-1) ** 0.5
    affected = (dist > threshold) * torch.isfinite(dist)
    xyz[affected] = xyz[affected] * (
        threshold * (2.0 - threshold / dist[affected]) / dist[affected]
    )[..., None]
    return xyz


def preprocess_img(x):
    if x.shape[2] != 224:
        assert x.shape[2] == 800
        x = F.interpolate(
            x,
            scale_factor=224./800.,
            mode="bilinear",
        )
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape((1, 3, 1, 1))
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((1, 3, 1, 1))
    imgs_normed = (x - resnet_mean) / resnet_std
    return imgs_normed


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

if __name__ == "__main__":
    # unit test
    '''
    test_patch_idx = torch.tensor([0,0,0,0,0,1,1,1,2,2,2,2,3,3])
    test_point_idx = torch.tensor([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3])
    attn = MCCDecoderAttention(1024)
    dummy_x = torch.zeros((4+29+14, 1024))
    attn(dummy_x, test_patch_idx, test_point_idx)
    '''