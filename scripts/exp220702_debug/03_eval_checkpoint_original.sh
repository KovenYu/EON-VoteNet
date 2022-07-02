#!/bin/bash
source /viscam/u/koven/anaconda3/bin/activate
conda activate votenet-3090
cd /viscam/u/koven/2022_EON/EON-VoteNet
LOGDIR='logs/exp220702_debug'
mkdir -p $LOGDIR
EXPDIR=$LOGDIR'/03_eval_checkpoint_original'
CUDA_VISIBLE_DEVICES=0 PYTHONBREAKPOINT='IPython.embed' python train.py --log_dir $EXPDIR \
--dataset_folder 'scan2cad_ex1.0_nosample' --is_eval --checkpoint_path '/viscam/u/koven/votenet/logs/exp211026_sunrgbd_estpose/10.1_estboth_scannet/checkpoint.tar'