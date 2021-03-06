# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import pc_util
from pointnet2_repo.pointnet2_modules import PointnetSAModuleVotes

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, mode_pose):
    mode_rot_mat = pc_util.batch_rotz(mode_pose)  # [B, Np, 3, 3]
    net_transposed = net.transpose(2,1) # (batch_size, 1024, 2+3+num_heading_bin*2+num_size_cluster*4)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    local_xyz_cano = net_transposed[:,:,2:5, None]
    local_xyz_world = torch.matmul(mode_rot_mat, local_xyz_cano).squeeze(-1)
    center = base_xyz + local_xyz_world # (batch_size, num_proposal, 3)
    end_points['center'] = center

    n_rot = num_heading_bin
    assert n_rot == num_heading_bin, 'our shifting algorithm assumes n_rot == n_heading_bin, but now n_rot=={}, n_heading_bin=={}'.format(n_rot, num_heading_bin)
    mode_pose_label, _ = pc_util.angle2class(mode_pose, n_rot)
    mode_pose_shift = mode_pose_label  # [B, Np]
    shifting = torch.arange(num_heading_bin)[None, None].to(mode_pose_shift.device)
    shifting = shifting.long() - mode_pose_shift[..., None].long()  # [B, Np, 24]
    shifting = shifting % num_heading_bin

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_scores = heading_scores.gather(2, shifting)

    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    heading_residuals_normalized = heading_residuals_normalized.gather(2, shifting)
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points, FLAGS=None):
        """
        Args:
            xyz: (B,N,3), in world space
            features: (B,C,N), in "canonical" space
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        seed_pose = end_points['point_pose_pred_angle']  # [B, N], already zeroed all bg points
        seed_mask = end_points['seed_mask_pred']  # [B, N]

        """you need seed_pose to 
        (1) back-rotate xyz to "canonical space" as pointnet input
        (2) rotate predicted bbox to world space"""
        # Farthest point sampling (FPS) on votes
        seed_pose_ = seed_pose
        xyz, features, fps_inds, mode_pose = self.vote_aggregation(xyz, features, seed_pose_, seed_mask)
        sample_inds = fps_inds
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,
                                   self.mean_size_arr, mode_pose)
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
