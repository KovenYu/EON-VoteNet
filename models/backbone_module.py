# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from pointnet2_repo.pointnet2_modules import *


def extract_canonical_feature(features, pose_labels, fg_mask=None, pool_method='max'):
    """
    Args:
        features:
            if (B, C, Nr, Np), return the mode feature within the Nr dimension, using
            the index provided by "poses". If (B, C, N), return itself without modification.
        pose_labels:
            (B, Np), ranging {0, 1, ..., Nr}
        fg_mask:
            if pool_method is not 'selection' then this should be [B, Np]
        pool_method:
            'mean', 'max', 'all_max' or 'selection'. Except for 'all_max',
            this is ONLY FOR BACKGROUND. For FG points you always use 'selection'
    """
    if len(features.shape) == 3: return features
    B, C, Nr, Np = features.shape
    if pool_method == 'selection':
        expanded_idx_array = pose_labels[..., None, None].expand(-1, -1, -1, C)  # [B, Np, 1, C]
        cano_feat = features.permute([0, 3, 2, 1]).gather(2, expanded_idx_array)  # [B, Np, 1, C],
        cano_feat = cano_feat.squeeze(2).permute([0, 2, 1])  # [B, C, Np]
        return cano_feat.contiguous()
    else:
        assert fg_mask is not None
        fg_features = features.permute([0, 3, 2, 1])[fg_mask]  # [Nfg, Nr, C]
        fg_pose_labels = pose_labels[fg_mask]  # [Nfg,]
        expanded_idx_array = fg_pose_labels[:, None, None].expand(-1, -1, C)  # [Nfg, 1, C]
        fg_cano_feat = fg_features.gather(1, expanded_idx_array).squeeze(1)  # [Nfg, C]

        bg_features = features.permute([0, 3, 2, 1])[~ fg_mask]  # [Nbg, Nr, C]
        if pool_method == 'mean':
            bg_cano_feat = bg_features.mean(dim=1)  # [Nbg, C]
        elif pool_method == 'max':
            bg_cano_feat = bg_features.max(dim=1)[0]  # [Nbg, C]
        else:
            assert False, 'unknown pool_method: {}'.format(pool_method)

        feat_holder = torch.zeros([B, Np, C], device=features.device)
        feat_holder[fg_mask] = fg_cano_feat
        feat_holder[~ fg_mask] = bg_cano_feat
        cano_feat = feat_holder.permute([0, 2, 1])
        return cano_feat


def construct_feature_orbit(features, Nr):
    """
    Args:
        features:
            if (B, C, Nr, Np), return itself without modification.
            if (B, C, Np),  expand it to (B, C, Nr, Np)
        Nr:
    """
    if len(features.shape) == 4: return features
    expanded_features = features[:, :, None, :].expand(-1, -1, Nr, -1)
    return expanded_features


class KPBackboneFullyConv(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB, 1 for height.
    """

    def __init__(self, norm_2d=False, n_rot=24, FLAGS=None):
        super().__init__()
        self.n_rot = n_rot
        self.FLAGS = FLAGS
        self.mask_net = MaskNet(in_dim=64, FLAGS=FLAGS)
        self.rot_attn_net = RotAttnNet(in_dim=64, FLAGS=FLAGS, n_cls=10 + 1)
        self.extract_canonical_feature = extract_canonical_feature

        self.sa1 = KPE2ModuleVotes(npoint=2048,
                            radius=0.2,
                            nsample=64,
                            in_dim=1,
                            out_dim=32,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            is_first_layer=True, )

        self.sa2 = KPE2ModuleVotes(npoint=1024,
                            radius=0.4,
                            nsample=32,
                            in_dim=32,
                            out_dim=32,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            )

        self.sa3 = KPE2ModuleVotes(npoint=512,
                            radius=0.8,
                            nsample=16,
                            in_dim=32,
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            )

        self.sa4 = KPE2ModuleVotes(npoint=256,
                            radius=1.2,
                            nsample=16,
                            in_dim=64,
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            )

        self.sa5 = KPE2ModuleVotes(npoint=128,
                            radius=1.5,
                            nsample=12,
                            in_dim=64,
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            )

        self.sa6 = KPE2ModuleVotes(npoint=64,
                            radius=1.8,
                            nsample=12,
                            in_dim=64,
                            out_dim=128,
                            norm_2d=norm_2d, n_rot=self.n_rot,
                            )

        self.fp1 = KPE2ModuleVotes(npoint=128,
                            radius=1.5,
                            nsample=12,
                            in_dim=(64 + 128),
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot, no_downsample=True,
                            )
        self.fp2 = KPE2ModuleVotes(npoint=256,
                            radius=1.2,
                            nsample=16,
                            in_dim=(64 + 64),
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot, no_downsample=True,
                            )
        self.fp3 = KPE2ModuleVotes(npoint=512,
                            radius=0.8,
                            nsample=16,
                            in_dim=(64 + 64),
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot, no_downsample=True,
                            )
        self.fp4 = KPE2ModuleVotes(npoint=1024,
                            radius=0.4,
                            nsample=32,
                            in_dim=(32 + 64),
                            out_dim=64,
                            norm_2d=norm_2d, n_rot=self.n_rot, no_downsample=True,
                            )
        print(self)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            pose_labels:
                [B, N], always 0 at bg points

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}

        xyz, features = self._break_up_pc(pointcloud)
        xyz, features, fps_inds, bq_idx = self.sa1(xyz, features)
        # [B, Np, 3], [B, C, Nr, Np], [B, Np] in {0,...,N}, [B, Np, Nnb] in {0,...N}

        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds, bq_idx = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds, bq_idx = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds, bq_idx = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        xyz, features, fps_inds, bq_idx = self.sa5(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa5_xyz'] = xyz
        end_points['sa5_features'] = features

        xyz, features, fps_inds, bq_idx = self.sa6(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa6_xyz'] = xyz
        end_points['sa6_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        skipped_feature = end_points['sa5_features']
        forward_feature = end_points['sa6_features']
        skipped_xyz = end_points['sa5_xyz']
        forward_feature = construct_feature_orbit(forward_feature, self.n_rot)
        skipped_feature = construct_feature_orbit(skipped_feature, self.n_rot)
        interpolated_feats = upsample_eqv_point_features(skipped_xyz, end_points['sa6_xyz'], forward_feature)
        cat_feats = torch.cat([interpolated_feats, skipped_feature], dim=1)
        _, features, _, bq_idx = self.fp1(skipped_xyz, cat_feats)

        skipped_feature = end_points['sa4_features']
        forward_feature = features
        skipped_xyz = end_points['sa4_xyz']
        forward_feature = construct_feature_orbit(forward_feature, self.n_rot)
        skipped_feature = construct_feature_orbit(skipped_feature, self.n_rot)
        interpolated_feats = upsample_eqv_point_features(skipped_xyz, end_points['sa5_xyz'], forward_feature)
        cat_feats = torch.cat([interpolated_feats, skipped_feature], dim=1)
        _, features, _, bq_idx = self.fp2(skipped_xyz, cat_feats)

        skipped_feature = end_points['sa3_features']
        forward_feature = features
        skipped_xyz = end_points['sa3_xyz']
        forward_feature = construct_feature_orbit(forward_feature, self.n_rot)
        skipped_feature = construct_feature_orbit(skipped_feature, self.n_rot)
        interpolated_feats = upsample_eqv_point_features(skipped_xyz, end_points['sa4_xyz'], forward_feature)
        cat_feats = torch.cat([interpolated_feats, skipped_feature], dim=1)
        _, features, _, bq_idx = self.fp3(skipped_xyz, cat_feats)

        skipped_feature = end_points['sa2_features']
        forward_feature = features
        skipped_xyz = end_points['sa2_xyz']
        forward_feature = construct_feature_orbit(forward_feature, self.n_rot)
        skipped_feature = construct_feature_orbit(skipped_feature, self.n_rot)
        interpolated_feats = upsample_eqv_point_features(skipped_xyz, end_points['sa3_xyz'], forward_feature)
        cat_feats = torch.cat([interpolated_feats, skipped_feature], dim=1)
        _, features, _, bq_idx = self.fp4(skipped_xyz, cat_feats)

        features_ = features  # [B, C, Nr, Np]
        seed_mask_logits, seed_mask_pred = self.mask_net(features_)
        point_cls_logits, point_cls_pred, point_rot_logits = self.rot_attn_net(features_)
        fg_mask = seed_mask_pred
        end_points['seed_mask_logits'] = seed_mask_logits
        end_points['seed_mask_pred'] = seed_mask_pred
        end_points['point_cls_logits'] = point_cls_logits
        end_points['point_cls_pred'] = point_cls_pred
        end_points['point_pose_logits'] = point_rot_logits  # [B, Nr, Np]
        end_points['point_pose_pred'] = point_rot_logits.argmax(dim=1)  # [B, Np]

        pose_for_selection = end_points['point_pose_pred']
        features_selected = self.extract_canonical_feature(features, pose_for_selection, fg_mask=torch.ones_like(fg_mask))

        features_pooled = self.extract_canonical_feature(features, pose_for_selection, fg_mask=torch.zeros_like(fg_mask))
        blending_weights = torch.softmax(seed_mask_logits, dim=2)  # [B, Np, 2]
        bg_weights = blending_weights[..., 0]
        fg_weights = blending_weights[..., 1]  # [B, Np]
        features = features_selected * fg_weights[:, None, :] + features_pooled * bg_weights[:, None, :]

        # print('after final shape:{}, {}'.format(end_points['sa2_xyz'].shape, features.shape))
        end_points['seed_features'] = features
        end_points['seed_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['seed_xyz'].shape[1]
        end_points['seed_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        return end_points


class RotAttnNet(nn.Module):
    def __init__(self, in_dim=256, FLAGS=None, n_cls=10+1):
        super(RotAttnNet, self).__init__()
        out_dim = n_cls*2
        self.net = pt_utils.SharedMLP([in_dim, 64, 64, out_dim], bn=True)
        self.Ncls = n_cls

    def forward(self, x):
        """
        Args:
            x: [B, C, Nr, Np]
        Returns:
            point_cls_logits:  [B, n_cls, Np]
            point_cls: [B, Np]
            point_rot_logits: [B, Nr, Np]
        """
        output = self.net(x)  # [B, N_cls*2, Nr, Np]
        B, _, Nr, Np = output.shape
        Ncls = self.Ncls
        point_logit_orbit = output[:, :Ncls, ...]  # [B, N_cls, Nr, Np]

        point_cls_logits = point_logit_orbit.mean(dim=2)  # [B, N_cls, Np]
        point_cls = point_cls_logits.argmax(dim=1)  # [B, Np]

        point_rot_logits_all = output[:, Ncls:2*Ncls, ...]  # [B, Ncls, Nr, Np]
        point_rot_logits = point_rot_logits_all.gather(1, point_cls[:, None, None, :].expand([-1, -1, Nr, -1]))  # [B,1,Nr,Np]
        point_rot_logits = point_rot_logits.squeeze(1)  # [B, Nr, Np]

        return point_cls_logits, point_cls, point_rot_logits


class MaskNet(nn.Module):
    def __init__(self, in_dim=64, FLAGS=None):
        super(MaskNet, self).__init__()
        self.FLAGS = FLAGS
        self.net = pt_utils.SharedMLP([in_dim, 64, 64, 2])
        
    def forward(self, x):
        """
        Args:
            x: [B, C, Nr, Np]
        Returns:
            mask_logits: [B, Np, 2], where 0-th logit for bg and 1-st logit for fg
            mask_pred: [B, Np], where 0 means bg, 1 means fg
        """
        B, _, _, Np = x.shape

        output = self.net(x)  # [B, 2, 1, Np]
        merged = output.mean(dim=2)
        mask_logits = merged.permute([0, 2, 1])  # [B, Np, 2]

        with torch.no_grad():
            mask_pred = mask_logits.argmax(dim=2) > 0.5  # [B, Np], Bool
        return mask_logits, mask_pred


if __name__ == '__main__':
    pass
