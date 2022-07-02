import sys
assert sys.version_info >= (3, 5)

import numpy as np
import scipy
import pathlib
import os
import JSONHelper
import quaternion
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import argparse
from scipy.spatial.transform import Rotation as R
import semantic_map
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from sunrgbd import sunrgbd_utils
from utils import pc_util
# params
parser = argparse.ArgumentParser()                                                                                                                                                                                                                                                                                        
parser.add_argument('--out', default="./meshes/", help="outdir")
parser.add_argument('--dataset_path', default="scan2cad_detection_labels")
parser.add_argument('--size_expansion', default=1.0, type=float)
opt = parser.parse_args()
opt.dataset_path = os.path.join(BASE_DIR, opt.dataset_path)
os.makedirs(opt.dataset_path, exist_ok=True)

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    # r = R.from_matrix(rot1[:3, :3])
    # angles = r.as_euler('yzx', degrees=True)
    # print('yzx angles: {}'.format(angles))
    return M

def find_TRS(composite_matrix):
    """
    Composite_matrix (denoted by M) is a 4x4 matrix from CAD space (+y up gravity, +z inward, +x left)
    to align space (+z up gravity, +y inward, +x right).
    It can transform a unit box ([1,1,1] and [-1,-1,-1] as diagonal points)
    to the bbox of an object to the align space.

    We want to find (1) the center of the obj, (2) the rotation of the obj w.r.t. a canonical pose,
    and (3) the XYZ scale of the obj. All represented in the align space.

    The idea is constructing some unit vectors to trace the transform.
    (1) center is found by Mc, where c=[0,0,0] in CAD space
    (2) length by 2||Mx-Mc||, where x=[-1,0,0], and similarly for y and z.
    (3) rotation by Mx-Mc, My-Mc, Mz-Mc, and construct rotation matrix from them.
    """
    x_axis = np.array([-1, 0, 0, 1])  # in the CAD space, which vector corresponds to your desired canonical x-axis in target space?
    y_axis = np.array([0, 0, 1, 1])
    z_axis = np.array([0, 1, 0, 1])
    center = np.array([0, 0, 0, 1])
    x_axis_after = composite_matrix.dot(x_axis)
    y_axis_after = composite_matrix.dot(y_axis)
    z_axis_after = composite_matrix.dot(z_axis)
    center_after = composite_matrix.dot(center)
    x_vector = (x_axis_after - center_after)[:3]
    y_vector = (y_axis_after - center_after)[:3]
    z_vector = (z_axis_after - center_after)[:3]
    x_normed = x_vector / np.linalg.norm(x_vector)
    y_normed = y_vector / np.linalg.norm(y_vector)
    z_normed = z_vector / np.linalg.norm(z_vector)

    rotation_mat = np.array([x_normed, y_normed, z_normed]).transpose()

    length_x = 2* np.linalg.norm(x_vector)
    length_y = 2 * np.linalg.norm(y_vector)
    length_z = 2 * np.linalg.norm(z_vector)
    length = np.array([length_x, length_y, length_z])
    return center_after[:3], rotation_mat, length

if __name__ == '__main__':
    save_mesh = False
    filename_json = os.path.join(BASE_DIR, "full_annotations.json")

    all_obbs = []

    for counter, r in enumerate(JSONHelper.read(filename_json)):
        id_scan = r["id_scan"]
        print('{}: {}'.format(counter, id_scan))  # something like "scene0470_00"

        outdir = os.path.abspath(opt.out + "/" + id_scan)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

        meta_file = '../scannet/scans' + "/" + id_scan + "/" + id_scan + ".txt"  # includes axisAlignment info for the train set scans.
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        scan2align = np.array(axis_align_matrix).reshape((4, 4))
        scan2world = make_M_from_tqs(r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"])
        world2scan = np.linalg.inv(scan2world)

        scannet_dataset_path = '../scannet/scannet_train_detection_data/'
        mesh_vertices = np.load(scannet_dataset_path + id_scan + '_vert.npy')
        semantic_labels = np.load(scannet_dataset_path + id_scan + '_sem_label.npy')  # NYU40
        point_cloud = mesh_vertices[:, 0:3]
        point_cloud_colors = mesh_vertices[:, 3:6]
        N = point_cloud.shape[0]

        if save_mesh:
            scan_file = meta_file.replace('.txt', "_vh_clean_2.ply")
            with open(scan_file, 'rb') as read_file:
                mesh_scan = PlyData.read(read_file)
            for v in mesh_scan["vertex"]:
                v1 = np.array([v[0], v[1], v[2], 1])
                v1 = np.dot(scan2align, v1)
                v[0] = v1[0]
                v[1] = v1[1]
                v[2] = v1[2]
                # <-- ignore normals etc.
            with open(outdir + "/scan_align.ply", mode='wb') as f:
                PlyData(mesh_scan).write(f)

        obbs_9d = []
        point_masks = np.zeros([N,]) > 1  # all False
        vote_masks = np.zeros([N,]) > 1  # all False
        point_cls = - np.ones([N,], dtype=np.int32)
        point_inst = - np.ones([N,], dtype=np.int32)
        point_votes = np.zeros([N, 9])  # 3x3 votes
        point_pose = np.zeros([N, 1])  # 3 poses
        point_sym = np.zeros([N, 1])
        point_vote_idx = np.zeros([N,]).astype(np.int32)  # in the range of {0,1,2}
        for model_counter, model in enumerate(r["aligned_models"]):

            cad2world = calc_Mbbox(model)
            cad2align = scan2align.dot(world2scan).dot(cad2world)
            center, rotation_cad2align, length = find_TRS(cad2align)
            length[length < 1e-2] = 1e-2
            r = R.from_matrix(rotation_cad2align)
            angles = r.as_euler('zyx', degrees=False)
            np.set_printoptions(precision=3)
            s = model['sym']
            if s == '__SYM_NONE':
                sym_cls, sym_idx = 'none', 0
            elif s == '__SYM_ROTATE_UP_2':
                sym_cls, sym_idx = 'sym2', 1
            elif s == '__SYM_ROTATE_UP_4':
                sym_cls, sym_idx = 'sym4', 2
            elif s == '__SYM_ROTATE_UP_INF':
                sym_cls, sym_idx = 'inf', 3
            else:
                raise Exception
            wnid = model['catid_cad']
            sem_cls_idx, sem_cls_name = semantic_map.wnid2label(wnid), semantic_map.wnid2name(wnid)
            obb_9d = np.hstack([center, length, angles[0], sym_idx, sem_cls_idx])  # save z-angle measured by radian
            z_angle_degree, z_angle_radian = angles[0] / 2 / np.pi * 360, angles[0]  # convert to degree
            # print('z angle: {}, center: {}, shape: {}, sym_cls: {}, sem_cls: {}'.format(z_angle_degree.astype(np.int32), center, length, sym_cls, sem_cls_name))

            # select pts inside the OBB
            corner_pts = sunrgbd_utils.my_compute_box_3d(center, length/2, -z_angle_radian)
            try:
                pc_in_obb, inds = sunrgbd_utils.extract_pc_in_box3d(point_cloud, corner_pts)  # S pts in this OBB, [S, 3], [S,]
            except ValueError:  # some very thin objects can lead to NAN when Euler angle transform
                continue
            vote_masks[inds] = True
            votes = center[None, ...] - pc_in_obb  # [S, 3]
            sparse_inds = np.arange(N)[inds]  # dense T/F -> sparse 0/1/2...;
            for i, sparse_ind in enumerate(sparse_inds):
                start = int(point_vote_idx[sparse_ind]) * 3
                point_votes[sparse_ind, start: start + 3] = votes[i]
                if start == 0:
                    point_votes[sparse_ind, 3:6] = point_votes[sparse_ind, 6:9] = votes[i]
            point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)

            # for pose and mask, expand it
            corner_pts = sunrgbd_utils.my_compute_box_3d(center, length/2*opt.size_expansion, -z_angle_radian)
            try:
                pc_in_obb, inds = sunrgbd_utils.extract_pc_in_box3d(point_cloud, corner_pts)  # S pts in this OBB, [S, 3], [S,]
            except ValueError:  # some very thin objects can lead to NAN when Euler angle transform
                continue
            point_masks[inds] = True
            point_cls[inds] = sem_cls_idx
            point_inst[inds] = model_counter
            sparse_inds = np.arange(N)[inds]  # dense T/F -> sparse 0/1/2...;
            for i, sparse_ind in enumerate(sparse_inds):
                point_pose[sparse_ind, 0] = z_angle_radian
                point_sym[sparse_ind, 0] = sym_idx

            if save_mesh:
                obj_name = '{}_{}'.format(model_counter, z_angle_degree.astype(np.int32))
                savename = outdir + "/bbox_{}.ply".format(obj_name)
                pc_util.my_write_bbox(savename, transform=cad2align)
                print('saved: {}'.format(savename))

                sem_labels_nyu40 = semantic_labels[inds]
                pc_util.write_ply_color(pc_in_obb, sem_labels_nyu40, os.path.join(outdir, "semPC_{}.ply".format(obj_name)))
                print('saved: {}'.format("semPC_{}.ply".format(obj_name)))

            obbs_9d.append(obb_9d)

        bg_pts = point_cloud[~ point_masks]
        fg_pts = point_cloud[point_masks]
        bg2fg_dist = scipy.spatial.distance_matrix(bg_pts, fg_pts)
        idx = np.argmin(bg2fg_dist, axis=1)
        point_pose[~ point_masks] = point_pose[point_masks].take(idx, axis=0)
        point_sym[~ point_masks] = point_sym[point_masks].take(idx, axis=0)

        obbs_9d = np.vstack(obbs_9d)
        all_obbs.append(obbs_9d)

        assert not np.any(np.isnan(obbs_9d)) and not np.any(np.isinf(obbs_9d))
        assert not np.any(np.isnan(point_votes)) and not np.any(np.isinf(point_votes))
        assert not np.any(np.isnan(point_masks)) and not np.any(np.isinf(point_masks))
        assert not np.any(np.isnan(point_pose)) and not np.any(np.isinf(point_pose))
        np.save('{}/{}_bbox.npy'.format(opt.dataset_path, id_scan), obbs_9d)
        np.save('{}/{}_pc.npy'.format(opt.dataset_path, id_scan), point_cloud)
        np.save('{}/{}_pc_color.npy'.format(opt.dataset_path, id_scan), point_cloud_colors)
        np.savez_compressed('{}/{}_pts_labels.npz'.format(opt.dataset_path, id_scan), point_inst=point_inst,
                            point_votes=point_votes, vote_masks=vote_masks, point_masks=point_masks, point_pose=point_pose, point_sym=point_sym, point_cls=point_cls)
        # print('No INF or NAN. Saved to {}/{}_bbox.npy and pts_labels.npz'.format(opt.dataset_path, id_scan))
    all_obbs = np.vstack(all_obbs)
    unique_cls = np.unique(all_obbs[:, 8])
    mean_sizes = []
    for cls_idx in unique_cls:
        idx = all_obbs[:, 8] == cls_idx
        obbs_this_cls = all_obbs[idx]  # [N_c, 9]
        mean_size = obbs_this_cls[:, 3:3+3].mean(axis=0)
        mean_sizes.append(mean_size)
    mean_sizes = np.vstack(mean_sizes)
    np.save('{}/mean_size.npy'.format(opt.dataset_path), mean_sizes)
    print('mean_sizes: {}'.format(mean_sizes))
