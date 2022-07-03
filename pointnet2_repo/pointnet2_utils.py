# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import pc_util
import pytorch3d.ops as pt3d_ops
from pointnet2_ops.pointnet2_utils import ball_query as bq


class QueryAndGroupPN(nn.Module):
    """
    NOTICE: this class is only used for aggregation (proposal) PointNet after voting.
    """
    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroupPN, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3), in world
        new_xyz : torch.Tensor
            centriods (B, npoint, 3), in world
        features : torch.Tensor
            Descriptors of the features (B, C, N), in canonical

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor, in canonical
        """
        idx = bq(self.radius, self.nsample, xyz, new_xyz).long()

        grouped_xyz = pt3d_ops.knn_gather(xyz, idx)  # [B,Np,Nnb,3]
        grouped_xyz = grouped_xyz.permute([0, 3, 1, 2])  # [B,3,Np,Nnb]
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        grouped_features = pt3d_ops.knn_gather(features.permute([0,2,1]), idx)  # [B,Np, Nnb, C]
        grouped_features = grouped_features.permute([0,3,1,2])  # [B,C,Np,Nnb]

        xyz_feature = grouped_xyz / self.radius

        new_features = torch.cat(
            [xyz_feature, grouped_features], dim=1  # the filter need not to know abs pts scale because that info has been provided by its layer index
        )  # (B, C + 3, npoint, nsample)

        return new_features


class QueryAndGroupKP(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroupKP, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = True

    def forward(self, xyz, new_xyz, features):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = bq(self.radius, self.nsample, xyz, new_xyz).long()  # [B, N_pts, N_nb]
        bq_idx = idx.clone()

        # create shadow feature idx
        N = xyz.shape[1]
        idx_ = idx.permute([2, 0, 1])  # [N_nb, B, N_pts]
        first_idxs = idx_[0]  # [B, N_pts]
        idx_zeroed = idx_ - first_idxs
        duplica_ind = idx_zeroed == 0
        duplica_ind[0] = False
        duplica_ind_ = duplica_ind.permute([1, 2, 0])
        idx[duplica_ind_] = N  # [B, Np, Nnb]

        shadow_xyz = 1e6 * torch.ones_like(xyz[:, 0:1, :])
        xyz = torch.cat([xyz, shadow_xyz], dim=1)  # (B,N+1,3)
        grouped_xyz = pt3d_ops.knn_gather(xyz, idx)  # [B,Np,Nnb,3]
        grouped_xyz = grouped_xyz.permute([0, 3, 1, 2])  # [B,3,Np,Nnb]
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        shadow_features = torch.zeros_like(features[:, :, 0:1])
        features = torch.cat([features, shadow_features], dim=-1)  # (B,C,N+1)
        grouped_features = pt3d_ops.knn_gather(features.permute([0, 2, 1]), idx)  # [B,Np, Nnb, C]
        grouped_features = grouped_features.permute([0, 3, 1, 2])  # [B,C,Np,Nnb]
        if self.use_xyz:
            xyz_feature = grouped_xyz.clone()
            xyz_feature[xyz_feature > 1e5] = 0
            xyz_feature = xyz_feature / self.radius
            new_features = torch.cat(
                [xyz_feature, grouped_features], dim=1  # the filter need not to know abs pts scale because that info has been provided by its layer index
            )  # (B, C + 3, npoint, nsample)
        else:
            new_features = grouped_features

        return new_features, grouped_xyz, bq_idx


class QueryAndGroupEquiv(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroupEquiv, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, Np, 3)
        features : torch.Tensor
            (B, C, Nr, N) where Nr is #rot, N is #input_pts

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, Nr, Np, Nnb) tensor
        grouped_xyz: (B,3,Np,Nnb)
        bq_idx: [B, Np, Nnb] ranging {0, 1, ..., Np-1}
        """
        idx = bq(self.radius, self.nsample, xyz, new_xyz).long()  # [B, N_pts, N_nb]
        bq_idx = idx.clone()

        # create shadow feature idx
        N = xyz.shape[1]
        idx_ = idx.permute([2, 0, 1])  # [N_nb, B, N_pts]
        first_idxs = idx_[0]  # [B, N_pts]
        idx_zeroed = idx_ - first_idxs
        duplica_ind = idx_zeroed == 0
        duplica_ind[0] = False
        duplica_ind_ = duplica_ind.permute([1, 2, 0])
        idx[duplica_ind_] = N  # [B, Np, Nnb]

        shadow_xyz = 1e6 * torch.ones_like(xyz[:, 0:1, :])
        xyz = torch.cat([xyz, shadow_xyz], dim=1)  # (B,N+1,3)
        grouped_xyz = pt3d_ops.knn_gather(xyz, idx)  # [B,Np,Nnb,3]
        grouped_xyz = grouped_xyz.permute([0, 3, 1, 2])  # [B,3,Np,Nnb]
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        shadow_features = torch.zeros_like(features[..., 0:1])
        features = torch.cat([features, shadow_features], dim=-1)  # (B,C,Nr,N+1)
        B, C, Nr, N1 = features.shape
        _, Np, Nnb = idx.shape
        features = features.flatten(start_dim=1, end_dim=2)  # [B,C*Nr,N+1]
        grouped_features = pt3d_ops.knn_gather(features.permute([0, 2, 1]), idx)  # [B,Np, Nnb, C*Nr]
        grouped_features = grouped_features.permute([0, 3, 1, 2])  # [B,C*Nr,Np,Nnb]
        grouped_features = grouped_features.view([B, C, Nr, Np, Nnb])
        if self.use_xyz:
            xyz_feature = grouped_xyz.clone()
            xyz_feature[xyz_feature > 1e5] = 0
            xyz_feature = xyz_feature / self.radius  # [B,3,Np,Nnb]
            new_features = torch.cat(
                [xyz_feature[:,:,None,...].expand([-1,-1,Nr,-1,-1]), grouped_features], dim=1  # the filter need not to know abs pts scale because that info has been provided by its layer index
            )  # (B, 3+C, Nr,Np,Nnb)
        else:
            new_features = grouped_features

        return new_features, grouped_xyz, bq_idx
