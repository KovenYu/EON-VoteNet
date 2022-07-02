# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Pointnet2 layers.
Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
Extended with the following:
1. Uniform sampling in each local region (sample_uniformly)
2. Return sampled points indices to support votenet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_
from kernel_points import load_kernels
import math
import numpy as np

pi = torch.tensor(np.pi)
from pytorch3d import ops as pt3d_ops
from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps


class PointnetSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,  # for RBF pooling
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz

        self.grouper = pointnet2_utils.QueryAndGroupPN(radius, nsample, use_xyz=use_xyz)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor,
                features, point_pose, point_pose_mask):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features, in world space
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features, in canonical space
        point_pose: [B, N] back-rotate xyz for each cluster, before feeding to pointNet
        point_pose_mask: [B, N] where 1 means FG, 0 means BG

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz, in world space
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors, in canonical space
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        mode_pose: [B, Np], mode pose of each cluster
        """
        B, N, _ = xyz.shape
        # batch_indictor = torch.arange(B)[..., None].expand([-1, N]).clone().cuda().flatten()  # [B*N]
        # xyz_flatten = xyz.flatten(0, 1)
        # inds = torch_cluster.fps(xyz_flatten, batch_indictor, ratio=self.npoint/N, random_start=False)  # [B*Np]
        # inds_ = inds[..., None].expand([-1, 3]).clone()
        # new_xyz = xyz_flatten.gather(0, inds_).view(B, self.npoint, 3)
        # inds = inds.view(B, self.npoint)
        # inds = inds - inds[..., 0:1]  # make the first inds zero
        # (B, N_out, 3)
        inds = fps(xyz, self.npoint).long()  # [B, Np]
        inds_ = inds[..., None].expand([-1, -1, 3]).clone()  # [B, Np, 3]
        new_xyz = xyz.gather(1, inds_)

        grouped_features, mode_pose = self.grouper(
            xyz, new_xyz, features, point_pose, point_pose_mask)
        # grouped_features: (B, C, Np, Nnb) where Nnb is padded by 0, and first 3-dim is back-rotated by mode_pose
        # mode_pose: [B, Np]

        new_features = self.mlp_module(
            grouped_features
        )  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_xyz, new_features, inds, mode_pose


class KPModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True,
            in_dim=1,
            out_dim=128,
            no_downsample=False,  # if no downsample, do conv on every point.
            is_first_layer=False
    ):
        super().__init__()

        self.is_first_layer = is_first_layer
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.in_dim = in_dim + 3 if use_xyz else in_dim
        self.no_downsample = no_downsample

        self.grouper = pointnet2_utils.QueryAndGroupKP(radius, nsample, use_xyz=use_xyz)
        self.kpconv = KPConv(15, 3, self.in_dim, out_dim, radius / 2.5, radius)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                pose_labels=None,
                fg_mask=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        pose_labels: None. This is just a placeholder. Not used.

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        bq_idx:
            [B, Np, Nnb]
        """

        if self.no_downsample:
            inds = None
            new_xyz = xyz
        else:
            if self.is_first_layer:
                inds = fps(xyz, self.npoint).long()  # [B, Np]
            else:
                B, Np = xyz.shape[0], self.npoint
                inds = torch.arange(Np)[None, :].expand([B, -1]).cuda()
            inds_ = inds[..., None].expand([-1, -1, 3]).clone()  # [B, Np, 3]
            new_xyz = xyz.gather(1, inds_)

        grouped_features, grouped_xyz, bq_idx = self.grouper(
            xyz, new_xyz, features
        )  # (B, D_in+3, Np, Nnghb), (B,3,Np,Nnghb), (B,Np)
        # now non-unique xyz has been shadowed (>1e6), non-unique features has been 0

        new_features = self.kpconv(grouped_features, grouped_xyz)  # (B, D_out, Np)
        new_features = self.leaky_relu(self.batch_norm(new_features))

        return new_xyz, new_features, inds, bq_idx


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', aggregation_mode='sum'):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.aggregation_mode = aggregation_mode

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, grouped_features, grouped_xyz):
        """

        Args:
            grouped_features: (B, D_in, Np, Nnghb), already shadowed
            grouped_xyz: (B,3,Np,Nnghb), already shadowed

        Returns:
            new_features, (B, D_out, Np)
        """

        # grouped_xyz (B,3,Np,Nnb) x kernel_points (K,3) -> dists (K,B,Np,Nnb)
        grouped_xyz_ = grouped_xyz.permute([0, 2, 3, 1]).contiguous()[None]  # (1,B,Np,Nnb,3)  # why 1e+6 becomes 0?
        kernel_pos = self.kernel_points[:, None, None, None, ...]  # (K,1,1,1,3)
        dists = (kernel_pos - grouped_xyz_).pow(2).sum(dim=-1).sqrt()

        # dists -> weights (K,B,Np,Nnb)
        weights = torch.clamp(1 - dists / self.KP_extent, min=0.0)
        # weights_sum = weights.sum(dim=0)

        # weights (K,B,Np,Nnb) x grouped_feat (B, Din, Np, Nnb) x kernels (K,Din,Dout) -> new_feat (B,Dout,Np)
        new_features = torch.einsum('kbnm,bdnm,kdf->bfn', weights, grouped_features, self.weights)

        return new_features

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class KPE2ModuleVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(
            self,
            *,
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True,
            in_dim=1,
            out_dim=128,
            n_rot=24,
            norm_2d=False,  # if True, normalize over both C and Nr; otherwise, normalize over C
            no_downsample=False,  # if no downsample, do conv on every point.
            is_first_layer=False
    ):
        super().__init__()

        self.is_first_layer = is_first_layer
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.in_dim = in_dim + 3 if use_xyz else in_dim
        self.n_rot = n_rot
        self.norm_2d = norm_2d
        self.no_downsample = no_downsample

        self.grouper = pointnet2_utils.QueryAndGroupEquiv(radius, nsample, use_xyz=use_xyz)
        self.kpconv = KPConvEquivSO2(15, 3, self.in_dim, out_dim, radius / 2.5, radius, compensate_xyz_feat=use_xyz,
                                     n_rot=self.n_rot)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.batch_norm = nn.BatchNorm1d(out_dim * n_rot) if norm_2d else nn.BatchNorm2d(out_dim)
        self.group_conv = GroupConvSO2(n_ch=out_dim)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, Nr, N) or (B, C, N) which is first layer

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, D_out, N_rot, npoint) tensor of the new_features descriptors
        inds:
            [B, Np], ranging [0,...,N]
        bq_idx:
            [B, Np, Nnb], ranging [0,...,N]
        """
        if len(features.shape) == 3:  # first layer; augment it
            features = features[:, :, None, :].expand([-1, -1, self.n_rot, -1])  # [B,C,Nr,N]

        if self.no_downsample:
            inds = None
            new_xyz = xyz
        else:
            if self.is_first_layer:
                inds = fps(xyz, self.npoint).long()  # [B, Np]
            else:
                B, Np = xyz.shape[0], self.npoint
                inds = torch.arange(Np)[None, :].expand([B, -1]).cuda()
            inds_ = inds[..., None].expand([-1, -1, 3]).clone()  # [B, Np, 3]
            new_xyz = xyz.gather(1, inds_)

        grouped_features, grouped_xyz, bq_idx = self.grouper(
            xyz, new_xyz, features
        )  # (B, 3+C, Nr, Np, Nnb), (B,3,Np,Nnb), [B, Np, Nnb]
        # now non-unique xyz has been shadowed (>1e6),
        # non-unique features (including the xyz in the first 3 dim) has been 0
        new_features = self.kpconv(grouped_features, grouped_xyz)  # (B, D_out, Nr, Np)
        if self.norm_2d:
            B, Dout, Nr, Np = new_features.shape
            new_features = new_features.flatten(1, 2)
            new_features = self.leaky_relu(self.batch_norm(new_features))
            new_features = new_features.view([B, Dout, Nr, Np])
        else:
            new_features = self.leaky_relu(self.batch_norm(new_features))
        new_features = self.group_conv(new_features)

        return new_xyz, new_features, inds, bq_idx


class KPConvEquivSO2(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius, compensate_xyz_feat=False,
                 fixed_kernel_points='center', aggregation_mode='sum', n_rot=24):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param compensate_xyz_feat: if True, rotate the first 3 feature dim of all input points because they are xyz
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param n_rot: number of bins to uniformally slice 2pi
        """
        super(KPConvEquivSO2, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.aggregation_mode = aggregation_mode
        self.n_rot = n_rot
        self.compensate_xyz_feat = compensate_xyz_feat

        # Initialize weights
        self.kernels = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_pos, self.inv_rot_mat = self.init_KP()  # [Nr, K, 3], [Nr, 3,3]

        return

    def reset_parameters(self):
        kaiming_uniform_(self.kernels, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)  # [K, 3]
        rot_mats = []
        for r in torch.arange(self.n_rot):
            rot_angle = r.float() / self.n_rot * 2 * pi
            rot_mat_2d = torch.tensor([[torch.cos(rot_angle), - torch.sin(rot_angle)],
                                       [torch.sin(rot_angle), torch.cos(rot_angle)]])
            rot_mat_3d = torch.eye(3)
            rot_mat_3d[:2, :2] = rot_mat_2d
            rot_mats.append(rot_mat_3d)
        rot_mats = torch.stack(rot_mats, dim=0)  # [Nr, 3, 3]

        kernel_pts = torch.tensor(K_points_numpy, dtype=torch.float32)[..., None]  # [K, 3, 1]
        group_kernel_pts = torch.matmul(rot_mats[:, None, ...], kernel_pts)  # [Nr, K, 3, 1]
        grp_knl_pts = Parameter(group_kernel_pts.squeeze(), requires_grad=False)  # [Nr, K, 3]

        # inverse-diretional rotate matrices for compensating first 3 feature dim
        inv_rot_mats = torch.inverse(rot_mats)
        inv_rot_mats = Parameter(inv_rot_mats, requires_grad=False)  # [Nr,3,3]

        return grp_knl_pts, inv_rot_mats

    def forward(self, grouped_features, grouped_xyz):
        """

        Args:
            grouped_features: (B, D_in, Nr, Np, Nnb), already shadowed, but not xyz-feat-compensated
            grouped_xyz: (B,3,Np,Nnb), already shadowed

        Returns:
            new_features, (B, D_out, Nr, Np)
        """
        # compensate xyz feature, if the first 3 dim is xyz feature
        if self.compensate_xyz_feat:
            xyz_feats = grouped_features[:, :3, ...]  # (B, 3, Nr, Np, Nnb)
            xyz_feats_ = xyz_feats.permute([0, 3, 4, 2, 1])[..., None]  # (B, Np, Nnb, Nr, 3, 1)
            xyz_feats_comp = torch.matmul(self.inv_rot_mat, xyz_feats_)  # (B, Np, Nnb, Nr, 3, 1)
            xyz_feats_comp = xyz_feats_comp.squeeze(-1)  # (B, Np, Nnb, Nr, 3)
            xyz_feats_comp = xyz_feats_comp.permute([0, 4, 3, 1, 2])  # (B, 3, Nr, Np, Nnb)
            grouped_features = torch.cat([xyz_feats_comp, grouped_features[:, 3:, ...]],
                                         dim=1)  # (B, D_in, Nr, Np, Nnb)

        # grouped_xyz (B,3,Np,Nnb) x kernel_pos (Nr,K,3) -> dists (Nr,K,B,Np,Nnb)
        grouped_xyz_ = grouped_xyz.permute([0, 2, 3, 1]).contiguous()[None]  # (1,B,Np,Nnb,3)
        kernel_pos = self.kernel_pos[..., None, None, None, :]  # (Nr,K,1,1,1,3)
        dists = (kernel_pos - grouped_xyz_).pow(2).sum(dim=-1).sqrt()

        # dists -> weights (Nr,K,B,Np,Nnb)
        weights = torch.clamp(1 - dists / self.KP_extent, min=0.0)

        # weights (Nr,K,B,Np,Nnb) x grouped_feat (B, Din, Nr, Np, Nnb) x kernels (K,Din,Dout) -> new_feat (B,Dout,Nr,Np)
        new_features = torch.einsum('rkbnm,bdrnm,kdf->bfrn', weights, grouped_features, self.kernels)

        return new_features

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)

class GroupConvSO2(nn.Module):
    def __init__(self, n_ch):
        super(GroupConvSO2, self).__init__()
        block = [nn.Conv1d(n_ch, n_ch, 3, padding=1, padding_mode='circular'),]
        block.append(nn.BatchNorm1d(n_ch))
        block.append(nn.LeakyReLU(0.1))
        self.net = nn.Sequential(*block)

    def forward(self, x):
        """
        Args:
            x: [B, C, Nr, Np] from point conv
        Returns:
            [B, C, Nr, Np]
        """
        B, C, Nr, Np = x.shape
        permuted = x.permute([0, 3, 1, 2])  # [B, Np, C, Nr]
        stacked = permuted.flatten(0, 1)  # [B*Np, C, Nr]
        conved = self.net(stacked)  # [B*Np, C, Nr]
        out = conved.view([B, Np, C, Nr]).permute([0, 2, 3, 1])
        return out


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another
    """

    def __init__(self, *, mlp, bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        interpolate m points to n>m points, and then concat the skip-linked points, and then MLP
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated, m<n

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        dist2, idx, _ = pt3d_ops.knn_points(unknown, known, K=3, return_sorted=False, return_nn=False)
        # [B, n, K=3], [B, n, K=3]
        dist_recip = 1.0 / (torch.sqrt(dist2) + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm  # [B, n, K=3],
        nn_features = pt3d_ops.knn_gather(known_feats.permute([0, 2, 1]), idx)  # [B, n, K=3, C2],

        interpolated_feats = (weight[..., None] * nn_features).sum(dim=2)  # [B, n, C2]
        interpolated_feats = interpolated_feats.permute([0, 2, 1])

        new_features = torch.cat([interpolated_feats, unknow_feats],
                                 dim=1)  # (B, C2 + C1, n)

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointnetEquivFPModule(nn.Module):

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        # now we assume bn is 1D, seeing rot-equiv as "data aug", so they should share the same bn
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        r"""
        interpolate m points to n>m points, and then concat the skip-linked points, and then MLP
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, Nr, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, Nr, m) tensor of features to be propigated, m<n

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], Nr, n) tensor of the features of the unknown features
        """
        B, C1, Nr, n = unknow_feats.shape
        _, C2, _, m = known_feats.shape

        dist2, idx, _ = pt3d_ops.knn_points(unknown, known, K=3, return_sorted=False, return_nn=False)
        # [B, n, K=3], [B, n, K=3]
        dist_recip = 1.0 / (torch.sqrt(dist2) + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm  # [B, n, K=3],

        known_feats = known_feats.flatten(1, 2).permute([0, 2, 1])  # [B, C2*Nr, m] -> [B, m, C2*Nr]
        nn_features = pt3d_ops.knn_gather(known_feats, idx)  # [B, n, K=3, C2*Nr],
        interpolated_feats = (weight[..., None] * nn_features).sum(dim=2)  # [B, n, C2*Nr]
        interpolated_feats = interpolated_feats.permute([0, 2, 1])

        interpolated_feats = interpolated_feats.view([B, C2, Nr, n])

        new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, Nr, n)

        new_features = self.mlp(new_features)

        return new_features


def upsample_point_features(target_position, source_position, source_feats):
    """
    interpolate m points to n>m points, and then concat the skip-linked points, and then MLP
    Parameters
    ----------
    target_position : torch.Tensor
        (B, n, 3) tensor of the xyz positions of the unknown features
    source_position : torch.Tensor
        (B, m, 3) tensor of the xyz positions of the known features
    source_feats : torch.Tensor
        (B, C, m) tensor of features to be propigated, m<n
    target_feats:
        (B, C_, n) if not None. Concated with interpolated features.

    Returns
    -------
    interpolated_feats : torch.Tensor
        (B, C, n) if target_feats is None, else (B, C+C_, n)
    """
    dist2, idx, _ = pt3d_ops.knn_points(target_position, source_position, K=3, return_sorted=False, return_nn=False)
    # [B, n, K=3], [B, n, K=3]
    dist_recip = 1.0 / (torch.sqrt(dist2) + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm  # [B, n, K=3],
    nn_features = pt3d_ops.knn_gather(source_feats.permute([0, 2, 1]), idx)  # [B, n, K=3, C],

    interpolated_feats = (weight[..., None] * nn_features).sum(dim=2)  # [B, n, C]
    interpolated_feats = interpolated_feats.permute([0, 2, 1])
    return interpolated_feats


def upsample_eqv_point_features(target_position, source_position, source_feats):
    """
    interpolate m points to n>m points, and then concat the skip-linked points, and then MLP
    Parameters
    ----------
    target_position : torch.Tensor
        (B, n, 3) tensor of the xyz positions of the unknown features
    source_position : torch.Tensor
        (B, m, 3) tensor of the xyz positions of the known features
    source_feats : torch.Tensor
        (B, C, Nr, m) tensor of features to be propigated, m<n

    Returns
    -------
    interpolated_feats : torch.Tensor
        (B, C, Nr, n)
    """
    n = target_position.shape[1]
    B, C, Nr, m = source_feats.shape

    dist2, idx, _ = pt3d_ops.knn_points(target_position, source_position, K=3, return_sorted=False, return_nn=False)
    # [B, n, K=3], [B, n, K=3]
    dist_recip = 1.0 / (torch.sqrt(dist2) + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm  # [B, n, K=3],

    source_feats = source_feats.flatten(1, 2).permute([0, 2, 1])  # [B, C*Nr, m] -> [B, m, C*Nr]
    nn_features = pt3d_ops.knn_gather(source_feats, idx)  # [B, n, K=3, C*Nr],
    interpolated_feats = (weight[..., None] * nn_features).sum(dim=2)  # [B, n, C*Nr]
    interpolated_feats = interpolated_feats.permute([0, 2, 1])
    interpolated_feats = interpolated_feats.view([B, C, Nr, n])

    return interpolated_feats


if __name__ == "__main__":
    pass
