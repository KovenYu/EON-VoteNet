# [CVPR22] Rotationally Equivariant 3D Object Detection
by [Hong-Xing Yu](https://kovenyu.com), [Jiajun Wu](https://jiajunwu.com/), and [Li Yi](https://ericyi.github.io/) from Stanford University and Tsinghua University.

arXiv link: [https://arxiv.org/abs/2204.13630](https://arxiv.org/abs/2204.13630) 

Project website: [https://kovenyu.com/EON](https://kovenyu.com/EON)

## Introduction

This repository is code release for our CVPR22 [paper](https://arxiv.org/pdf/2204.13630.pdf).

Since the proposed method is a plug-and-play design, in this repository we provide
an implementation on VoteNet based on its [official repository](https://github.com/facebookresearch/votenet) with some modifications.
For better understanding, we provide two branches.
The `baseline` branch provides the baseline VoteNet,
and the `main` branch provides EON-VoteNet with minimal modification. 

## Environment
This codebase has been tested under the following environment:
- Python 3.8.8
- pytorch 1.9.0, cuda 11.1, torchvision 0.10.0
- pointnet2_ops 3.0.0 and pointnet2 3.0.0 from https://github.com/erikwijmans/Pointnet2_PyTorch
- pytorch3d 0.5.0, numpy-quaternion
- (pip install) matplotlib, opencv-python, plyfile, 'trimesh>=2.35.39,<2.35.40','networkx>=2.2,<2.3'

## Datasets
Currently, our codebase only supports ScanNet V2 with Scan2CAD detection labels.
Download the ScanNet data following the [README in scannet folder](scannet/README.md).
Then,
```
cd EON-VoteNet/scan2cad
python generate_cad2align_and_rot.py
```
which generates scan2cad labels to `EON-VoteNet/scan2cad/scan2cad_detection_labels`. 

## Training
Training assumes one GPU.
It should take a few hours using the default setup.
```
cd EON-VoteNet
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Evaluation
After training or [downloading a checkpoint](https://drive.google.com/file/d/1d2k478v9UFkI_UwVbZ1345z46ddnrUMe/view?usp=sharing),
evaluation can be done by:
```
python train.py --is_eval --checkpoint_path [checkpoint_file_path]
```

## Citation
If you find our repository useful, please consider
citing our paper and also [VoteNet](https://github.com/facebookresearch/votenet#citation).
```
@inproceedings{yu2022eon,
    author = {Yu, Hong-Xing and Wu, Jiajun and Yi, Li},
    title = {Rotationally Equivariant 3D Object Detection},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},
    year = {2022}
}
```