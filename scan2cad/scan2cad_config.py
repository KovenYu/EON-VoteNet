# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
import semantic_map
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class Scan2CadDatasetConfig(object):
    def __init__(self, n_rot=24):
        self.num_class = self.num_size_cluster = 10
        self.num_heading_bin = n_rot
        """in order to easily rotate bbox, we now assume n_bins == n_rot"""

        self.type2class = semantic_map.OURS_NAME_TO_LABEL
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type_mean_size = {
            'chair': [0.55067555, 0.57861282, 0.84943989],
            'table': [1.24506055, 0.72456264, 0.66167052],
            'cabinet': [0.88929522, 0.56084465, 0.95581982],
            'bin': [0.36641968, 0.27876529, 0.45580824],
            'bookshelf': [1.05132405, 0.33744384, 1.3471979],
            'display': [0.60694304, 0.16549277, 0.474938],
            'sofa': [1.64314938, 0.8564234, 0.74599956],
            'bath': [0.85305382, 0.5161201, 0.43925024],
            'bed': [1.37070364, 2.06004587, 1.12196365],
            'others': [0.50858551, 0.45845599, 0.61929728]
        }
        name_list = []
        size_list = []
        for name, shape in self.type_mean_size.items():
            size = np.array(shape).prod()
            size_list.append(size)
            name_list.append(name)
        sizes = np.array(size_list)
        names = np.array(name_list, dtype=object)
        order = sizes.argsort()
        self.size_sorted_classes = names[order]

        self.mean_size_arr = np.zeros((len(self.type_mean_size), 3))
        for i in range(len(self.type_mean_size)):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb


