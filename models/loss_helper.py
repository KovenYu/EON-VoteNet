# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask'].float()
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_point_pose_loss(end_points):
    point_pose_logits = end_points['point_pose_logits']  # [B, Nr, Np=1024]
    point_pose_logits = point_pose_logits.permute([0, 2, 1]).flatten(0, 1)  # [B*Np, Nr]
    seed_pose_gt_label = end_points['seed_pose_gt_label'].flatten(0, 1)  # [B*Np]
    seed_gt_mask = end_points['seed_gt_mask'].flatten(0, 1)  # [B*Np], 1 means FG, 0 means BG
    seed_gt_mask = end_points['seed_mask_pred'].flatten(0, 1)
    seed_sym_label = end_points['seed_sym_label'].flatten(0, 1)  # [B*Np]

    Nr = point_pose_logits.shape[1]
    sym2_alter_pose_label = (seed_pose_gt_label + Nr // 2) % Nr
    sym2_mask = seed_sym_label == 1

    sym4_alter_pose_label_0 = (seed_pose_gt_label + Nr // 4 * 0) % Nr
    sym4_alter_pose_label_1 = (seed_pose_gt_label + Nr // 4 * 1) % Nr
    sym4_alter_pose_label_2 = (seed_pose_gt_label + Nr // 4 * 2) % Nr
    sym4_alter_pose_label_3 = (seed_pose_gt_label + Nr // 4 * 3) % Nr
    sym4_mask = (seed_sym_label == 2) | (seed_sym_label == 3)
    fg_mask = seed_gt_mask == True
    bg_mask = seed_gt_mask == False

    point_cls_logits = end_points['point_cls_logits']  # [B, N_cls, Np]
    point_cls_logits = point_cls_logits.permute([0, 2, 1]).flatten(0, 1)  # [B*Np, N_cls]
    seed_cls_gt_label = end_points['seed_cls_gt_label'].flatten(0, 1)  # [B*Np]
    seed_cls_gt_label = seed_cls_gt_label + 1  # now starting from 0

    criterion = nn.CrossEntropyLoss(reduction='none')
    pose_cls_loss_ = criterion(point_pose_logits, seed_pose_gt_label.long())
    pose_cls_loss_sym2_alter_ = criterion(point_pose_logits, sym2_alter_pose_label)
    sym2_loss_1 = pose_cls_loss_[sym2_mask]
    sym2_loss_2 = pose_cls_loss_sym2_alter_[sym2_mask]
    sym2_min_loss = torch.minimum(sym2_loss_1, sym2_loss_2)
    pose_cls_loss_[sym2_mask] = sym2_min_loss

    pose_cls_loss_sym4_alter_0 = criterion(point_pose_logits, sym4_alter_pose_label_0)[sym4_mask]
    pose_cls_loss_sym4_alter_1 = criterion(point_pose_logits, sym4_alter_pose_label_1)[sym4_mask]
    pose_cls_loss_sym4_alter_2 = criterion(point_pose_logits, sym4_alter_pose_label_2)[sym4_mask]
    pose_cls_loss_sym4_alter_3 = criterion(point_pose_logits, sym4_alter_pose_label_3)[sym4_mask]
    sym4_min_loss = torch.minimum(pose_cls_loss_sym4_alter_0, pose_cls_loss_sym4_alter_1)
    sym4_min_loss = torch.minimum(sym4_min_loss, pose_cls_loss_sym4_alter_2)
    sym4_min_loss = torch.minimum(sym4_min_loss, pose_cls_loss_sym4_alter_3)
    pose_cls_loss_[sym4_mask] = sym4_min_loss

    weight_vector = torch.zeros_like(fg_mask.float())
    weight_vector[fg_mask] = 1.
    weight_vector[bg_mask] = 0.
    point_pose_loss = torch.sum(pose_cls_loss_ * weight_vector) / (torch.sum(weight_vector) + 1e-6)

    point_cls_loss_ = criterion(point_cls_logits, seed_cls_gt_label)
    point_cls_loss = torch.sum(point_cls_loss_ * weight_vector) / (torch.sum(weight_vector) + 1e-6)

    with torch.no_grad():
        pose_acc_all = point_pose_logits.argmax(dim=1) == seed_pose_gt_label
        pose_acc_sym2_alter = point_pose_logits.argmax(dim=1) == sym2_alter_pose_label
        pose_acc_all[sym2_mask] = pose_acc_all[sym2_mask] | pose_acc_sym2_alter[sym2_mask]
        pose_acc_all[sym4_mask] = True
        pose_acc_fg = pose_acc_all[fg_mask].float().mean()
        pose_acc_bg = pose_acc_all[bg_mask].float().mean()
        pose_acc = pose_acc_all.float().mean()

    point_cls_acc_all = point_cls_logits.argmax(dim=1) == seed_cls_gt_label
    point_cls_acc_fg = point_cls_acc_all[fg_mask].float().mean()
    point_cls_acc_bg = point_cls_acc_all[bg_mask].float().mean()
    point_cls_acc = point_cls_acc_all.float().mean()

    end_points['point_pose_loss'] = point_pose_loss
    end_points['point_pose_acc'] = pose_acc
    end_points['point_pose_acc_fg'] = pose_acc_fg
    end_points['point_pose_acc_bg'] = pose_acc_bg

    end_points['point_cls_loss'] = point_cls_loss
    end_points['point_cls_acc'] = point_cls_acc
    end_points['point_cls_acc_fg'] = point_cls_acc_fg
    end_points['point_cls_acc_bg'] = point_cls_acc_bg

    return end_points, point_pose_loss, point_cls_loss, pose_acc_all


def compute_point_mask_loss(end_points, point_mask_logits, point_gt_mask, bg_weight=.25):
    """
    Args:
        end_points:
        point_mask_logits: [B, Np, 2]
        point_gt_mask: [B, Np]
        bg_weight:
    Returns:
    """
    point_mask_logits = point_mask_logits.flatten(0, 1)  # [B*Np, 2]
    point_gt_mask = point_gt_mask.flatten(0, 1).long()  # [B*Np]
    weight = torch.tensor([bg_weight, 1.]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)
    mask_loss = criterion(point_mask_logits, point_gt_mask)
    return end_points, mask_loss


def get_loss(end_points, config, FLAGS=None):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    seed_inds = end_points['seed_inds']  # [B, Np]
    seed_rot_gt_label = end_points['pose_label'].gather(1, seed_inds.long())  # [B, Np]
    seed_gt_mask = end_points['point_mask'].gather(1, seed_inds.long())  # [B, Np]
    seed_cls_gt_label = end_points['point_cls_label'].gather(1, seed_inds)
    seed_sym_label = end_points['sym_label'].gather(1, seed_inds)
    end_points['seed_pose_gt_label'] = seed_rot_gt_label
    end_points['seed_gt_mask'] = seed_gt_mask
    end_points['seed_cls_gt_label'] = seed_cls_gt_label
    end_points['seed_sym_label'] = seed_sym_label

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss
    end_points, point_pose_loss, point_cls_loss, pose_acc_all = compute_point_pose_loss(end_points)

    seed_gt_mask = end_points['seed_gt_mask']
    end_points, seed_mask_loss = compute_point_mask_loss(end_points, end_points['seed_mask_logits'], seed_gt_mask)
    end_points['seed_mask_loss'] = seed_mask_loss
    seed_mask_acc_all = seed_gt_mask == end_points['seed_mask_pred']
    end_points['seed_mask_acc'] = seed_mask_acc_all.float().mean()
    end_points['seed_mask_fg_acc'] = seed_mask_acc_all[seed_gt_mask].float().mean()
    end_points['seed_mask_bg_acc'] = seed_mask_acc_all[~ seed_gt_mask].float().mean()
    tp_fg = (seed_gt_mask & end_points['seed_mask_pred']).float().mean()
    fp_fg = ((~ seed_gt_mask) & end_points['seed_mask_pred']).float().mean()
    fn_fg = (seed_gt_mask & (~ end_points['seed_mask_pred'])).float().mean()
    iou_fg = tp_fg / (tp_fg + fp_fg + fn_fg)
    tp_bg = ((~ seed_gt_mask) & (~ end_points['seed_mask_pred'])).float().mean()
    fp_bg = (seed_gt_mask & (~ end_points['seed_mask_pred'])).float().mean()
    fn_bg = ((~ seed_gt_mask) & end_points['seed_mask_pred']).float().mean()
    iou_bg = tp_bg / (tp_bg + fp_bg + fn_bg)
    end_points['seed_fg_iou_acc'] = iou_fg
    end_points['seed_bg_iou_acc'] = iou_bg

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + 0.1*(point_cls_loss + point_pose_loss) + 0.1* seed_mask_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
