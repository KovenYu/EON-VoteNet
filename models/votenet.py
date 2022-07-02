# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import pc_util
from backbone_module import *
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results, dump_examples
from loss_helper import get_loss


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, n_rot=24,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', norm_2d=False, FLAGS=None):
        super().__init__()

        self.num_class = num_class
        self.n_rot = n_rot
        self.FLAGS = FLAGS
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = KPBackboneFullyConv(norm_2d=norm_2d, n_rot=n_rot,
                                                FLAGS=FLAGS)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, seed_feature_dim=64)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling, seed_feat_dim=64)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        # --------- HOUGH VOTING ---------
        xyz = end_points['seed_xyz']  # [B, Np, 3]
        features = end_points['seed_features']  # [B, C, Np]

        seed_rot_pred = pc_util.class2angle(end_points['point_pose_pred'].float(), 0, self.n_rot)
        seed_mask = end_points['seed_mask_pred']
        seed_rot_pred[~ seed_mask] = 0  # BG pose not defined, so you don't want it to rotate bg vote
        end_points['point_pose_pred_angle'] = seed_rot_pred
        vote_rotate = seed_rot_pred
        xyz, features = self.vgen(xyz, features, vote_rotate)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points, FLAGS=self.FLAGS)

        return end_points
