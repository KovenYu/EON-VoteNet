# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'scan2cad')
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import utils
from utils import pc_util
from sunrgbd import sunrgbd_utils
from scan2cad_config import Scan2CadDatasetConfig
MAX_NUM_OBJ = 64

class Scan2CadDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', num_points=40000, n_rot=4, dataset_folder='scan2cad_train_detection_data',
        use_height=True, augment=True, return_color=False):
        self.angle_alignment = 'forward aligns with negative y axis.'
        self.n_rot = n_rot
        self.DC = Scan2CadDatasetConfig(n_rot=self.n_rot)
        self.return_color = return_color
        self.data_path = os.path.join(DATA_DIR, dataset_folder)
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(DATA_DIR, 'meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
        else:
            print('illegal split name')
            return
        
        self.num_points = num_points
        self.use_height = use_height
        self.augment = augment
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sym_idx: (MAX_NUM_OBJ,) indicating symmetry order
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,9) with votes XYZ, three possible votes
            point_masks: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            point_pose_label: (N,3) 3 possible poses, discretized to labels
            point_pose_angle: (N,3) 3 possible poses, measured by radian [0, 2*pi]
            point_sym_label: (N,3) corresponding to each possible pose, truncated to .01 for easy comparison
            scan_idx: int scan index in scan_names list
        """
        
        scan_name = self.scan_names[idx]
        point_cloud = np.load('{}/{}_pc.npy'.format(self.data_path, scan_name))
        bboxes = np.load('{}/{}_bbox.npy'.format(self.data_path, scan_name))
        pts_labels = np.load('{}/{}_pts_labels.npz'.format(self.data_path, scan_name))
        point_cloud_color = np.load('{}/{}_pc_color.npy'.format(self.data_path, scan_name)).astype(np.float32) / 255
        point_votes, point_masks, point_pose, point_sym, point_cls  = pts_labels['point_votes'], pts_labels['point_masks'], pts_labels['point_pose'], pts_labels['point_sym'], pts_labels['point_cls']
        # point_pose and bboxes[:, 6] ranging [-pi, pi]
        try:
            vote_masks = pts_labels['vote_masks']
        except KeyError:
            vote_masks = point_masks

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = - bboxes[:, 6]
                point_votes[:, [0, 3, 6]] = -1 * point_votes[:, [0, 3, 6]]
                point_pose = - point_pose

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                bboxes[:, 1] = -1 * bboxes[:, 1]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                point_votes[:, [1, 4, 7]] = -1 * point_votes[:, [1, 4, 7]]
                point_pose = np.pi - point_pose

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:, :3] = np.dot(point_cloud[:, 0:3] + point_votes[:, :3], np.transpose(rot_mat))
            point_votes_end[:, 3:6] = np.dot(point_cloud[:, 0:3] + point_votes[:, 3:6], np.transpose(rot_mat))
            point_votes_end[:, 6:] = np.dot(point_cloud[:, 0:3] + point_votes[:, 6:], np.transpose(rot_mat))

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] += rot_angle
            point_pose += rot_angle
            point_votes[:, :3] = point_votes_end[:, :3] - point_cloud[:, 0:3]
            point_votes[:, 3:6] = point_votes_end[:, 3:6] - point_cloud[:, 0:3]
            point_votes[:, 6:] = point_votes_end[:, 6:] - point_cloud[:, 0:3]

        point_pose %= (2*np.pi)
        bboxes[:, 6] %= (2*np.pi)

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        semantic_classes = np.zeros((MAX_NUM_OBJ,))
        label_mask = np.zeros((MAX_NUM_OBJ)) > 1e10
        label_mask[0:bboxes.shape[0]] = True

        for i, bbox in enumerate(bboxes):
            semantic_class = bbox[8]
            semantic_classes[i] = semantic_class
            box3d_centers[i] = bbox[:3]
            angle_class, angle_residual = self.DC.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_class, size_residual = self.DC.size2class(bbox[3:6], self.DC.class2type[semantic_class])
            size_classes[i] = size_class
            size_residuals[i] = size_residual

        # using only the first for now; fix it later
        point_votes = np.tile(point_votes[:, :3], [1, 3])
        point_pose = point_pose[:, 0]
        point_sym = point_sym[:, 0]

        pose_label, pose_res = pc_util.angle2class(point_pose, self.n_rot)

        if point_cloud.shape[0] != self.num_points:
            point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
            point_masks, point_votes, pose_label, point_pose, point_sym, point_cls, vote_masks = point_masks[choices], point_votes[choices], pose_label[choices], point_pose[choices], point_sym[choices], point_cls[choices], vote_masks[choices]
            point_cloud_color = point_cloud_color[choices]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = box3d_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['sem_cls_label'] = semantic_classes.astype(np.int64)
        ret_dict['box_label_mask'] = label_mask
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_mask'] = vote_masks
        ret_dict['point_mask'] = point_masks
        ret_dict['pose_label'] = pose_label.astype(np.int64)
        ret_dict['pose_angle'] = point_pose.astype(np.float32)
        ret_dict['point_cls_label'] = point_cls.astype(np.int64)
        ret_dict['sym_label'] = point_sym.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['point_cloud_color'] = point_cloud_color
        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name='', save_path='meshes'):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds]
    for j in range(3):
        vote1 = point_votes[inds, j*3:(j+1)*3]
        center = pc_obj + vote1 if pc.shape[1] == 3 else pc_obj + np.hstack([vote1, np.zeros_like(vote1)])
        pc_to_save = np.vstack([pc_obj, center])
        n_p = pc_obj.shape[0]
        edges = [(i, i+n_p) for i in range(n_p)]
        pc_util.write_ply_edge(pc_to_save, edges, '{}/{}_pc_obj_voted{}.ply'.format(save_path, name, j))
    
def viz_obb(DC, pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name='', save_path='meshes', cls_label=None):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    K = label.shape[0]
    if cls_label is None:
        cls_label = np.zeros(K)
    for i in range(K):
        if not mask[i]: continue
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])  # [-pi, pi]
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        center = label[i]
        scale_mat = np.eye(4)
        scale_mat[:3, :3] = np.diag(box_size/2)
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = pc_util.rotz(heading_angle)
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = center
        transform = trans_mat.dot(rot_mat).dot(scale_mat)
        int_angle_degree = int(heading_angle/2/np.pi * 360)  # [-180, 180]
        pc_util.my_write_bbox('{}/{}_{}_gt_obbs_{}.ply'.format(save_path, name, i, int_angle_degree), transform=transform,
                              color_idx=cls_label[i])

def viz_pose(pc, pose_angle, pose_label, pose_mask, name='', save_dir='meshes'):
    """
    visualize colorful
    Args:
        pc: [N, 3 or 6]
        pose_angle: [N,] ranging [-pi, pi]
        pose_label: [N,]
        pose_mask: [N,] T or F
        name:
        save_dir:
    """
    pc = pc[pose_mask, :3]
    # angles = pose_angle[pose_mask] % (2*np.pi)
    angles = pc_util.class2angle(pose_label[pose_mask], 0, 24)
    angles = angles % (2*np.pi)
    angles_norm = angles / (2*np.pi)
    save_path = '{}/{}_pose.ply'.format(save_dir, name)
    pc_util.write_ply_color_continuous(pc, angles_norm, save_path)


def viz_fg_mask(pc, point_mask, name='', save_dir='meshes'):
    save_path = '{}/{}_fgmask.ply'.format(save_dir, name)
    pc_util.write_ply_color_continuous(pc, point_mask, save_path)
    
if __name__=='__main__':
    pass