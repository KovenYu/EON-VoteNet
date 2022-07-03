# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2_repo'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2_repo.pytorch_utils import BNMomentumScheduler
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='scan2cad', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--n_rot', type=int, default=4, help='Number of discretized rotation bins')
parser.add_argument('--is_eval', action='store_true')
parser.add_argument('--dataset_folder', default='scan2cad_detection_labels')
parser.add_argument('--nworkers', default=8, type=int)
FLAGS = parser.parse_args()
FLAGS.num_point = 20000 if FLAGS.dataset == 'sunrgbd' else 40000

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(LOG_DIR, 'dump_results')
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s' % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt' if FLAGS.is_eval else 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR, exist_ok=True)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Create Dataset and Dataloader
assert FLAGS.dataset == 'scan2cad', "Current implementation only supports ScanNet with Scan2CAD labels."
sys.path.append(os.path.join(ROOT_DIR, 'scan2cad'))
total_train_exmaples = 1193
from scan2cad.scan2cad_detection_dataset import Scan2CadDetectionDataset, MAX_NUM_OBJ
from scan2cad.scan2cad_config import Scan2CadDatasetConfig

DATASET_CONFIG = Scan2CadDatasetConfig(FLAGS.n_rot)
TRAIN_DATASET = Scan2CadDetectionDataset('train', num_points=NUM_POINT, dataset_folder=FLAGS.dataset_folder,
                                         augment=True, use_height=(not FLAGS.no_height),
                                         n_rot=FLAGS.n_rot)
TEST_DATASET = Scan2CadDetectionDataset('val', num_points=NUM_POINT, dataset_folder=FLAGS.dataset_folder,
                                        augment=False, use_height=(not FLAGS.no_height),
                                        n_rot=FLAGS.n_rot)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=FLAGS.nworkers, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=FLAGS.nworkers, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
from models import votenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

Detector = votenet.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling,
               FLAGS=FLAGS)

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)
criterion = votenet.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
iters_per_epoch = np.ceil(total_train_exmaples / BATCH_SIZE)
lr_decay_iters = [int(x * iters_per_epoch) for x in LR_DECAY_STEPS]
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_iters)

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
               'per_class_proposal': True, 'conf_thresh': 0.05,
               'dataset_config': DATASET_CONFIG}

if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    print('loading checkpoint from {}'.format(CHECKPOINT_PATH))
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {}  # collect statistics
    bnm_scheduler.step()  # decay BN momentum
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        ################## profile timing ####################
        # with torch.autograd.profiler.profile(use_cuda=True,record_shapes=True) as prof:
        #     end_points = net(inputs)
        # # print(prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total'))
        # print(prof.key_averages().table(sort_by='cuda_time_total'))
        # if batch_idx > 5:  # wait until stable
        #     import IPython
        #     IPython.embed()
        ################## end profile timing ####################
        end_points = net(inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points), '{} should not in end_points.'.format(key)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG, FLAGS=FLAGS)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
        scheduler.step()


import time
def evaluate_one_epoch(eval_few=False):
    stat_dict = {}  # collect statistics
    ap_calculator_list = [APCalculator(ap_iou_thresh=iou_thresh,
                                       class2type_map=DATASET_CONFIG.class2type) for iou_thresh in [0.25, 0.5]]
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if eval_few and (batch_idx >= len(TEST_DATALOADER) / 10):
            break
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            start = time.time()
            end_points = net(inputs)
            end = time.time() - start
            print('forward time: {}'.format(end))

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG, FLAGS=FLAGS)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    for ap_calculator in ap_calculator_list:
        metrics_dict = ap_calculator.compute_metrics()
        sorted_metric_keys = ['{} Average Precision'.format(x) for x in DATASET_CONFIG.size_sorted_classes] \
                             + ['mAP'] \
                             + ['{} Recall'.format(x) for x in DATASET_CONFIG.size_sorted_classes] \
                             + ['AR']
        for key in sorted_metric_keys:
            log_string('eval %s: %f' % (key, metrics_dict[key]))

        sorted_AP_keys = ['{} Average Precision'.format(x) for x in DATASET_CONFIG.size_sorted_classes] \
                         + ['mAP']
        result_table = ''
        for key in sorted_AP_keys:
            AP_this = metrics_dict[key]
            AP_this = AP_this * 100
            result_table += '{:.1f}\t'.format(AP_this)
        log_string('result_table: {}'.format(result_table))

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss

def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (scheduler.get_last_lr()[0]))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        if FLAGS.is_eval:
            evaluate_one_epoch()
            return
        train_one_epoch()
        is_test_epoch = (EPOCH_CNT % 20 == 19)
        if is_test_epoch:
            eval_few = ((EPOCH_CNT != MAX_EPOCH - 1) and FLAGS.dataset=='sunrgbd')
            loss = evaluate_one_epoch(eval_few=eval_few)
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    set_seed(0)
    train(start_epoch)