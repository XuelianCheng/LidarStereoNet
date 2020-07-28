## LidarStereoNet
This is the code for the paper:

Xuelian Cheng*, Yiran Zhong*, Yuchao Dai, Pan Ji, Hongdong Li.
_Noise-Aware Unsupervised Deep Lidar-Stereo Fusion._
In CVPR, 2019. https://arxiv.org/abs/1904.03868.

Watch our video: https://youtu.be/8wBzUY8bAvU.


### Citing
If you find this code useful, please cite our work.

```
@inproceedings{cheng2019cvpr,
title={Noise-Aware Unsupervised Deep Lidar-Stereo Fusion},
author={Cheng, Xuelian and Zhong, Yiran and Dai, Yuchao and Ji, Pan and Li, Hongdong},
booktitle={Proc. IEEE Conf. Comp. Vis. Patt. Recogn.},
year={2019}
}
```


## Model
We provide the checkpoints in [Best Model](https://drive.google.com/file/d/1NdEBdrUq8iM9ZkWjWmvfSph-3fPoE4yu/view?usp=sharing).
To access our results on the selected KITTI 141 dataset , please directly use this link [Inference Images](https://drive.google.com/file/d/1XnrEU6Xwsok20EdFoSswkmkgdx1dHNUy/view?usp=sharing) and use matlab code to compute error matrix.

```shell
run Inference_kitti141.m
```

## Installation

### Environment

1. Ubuntu 16.04 LTS
2. CUDA 10.0
3. PyTorch 1.0.0
4. TorchVision 0.4.0

### Install
Create a  virtual environment and activate it.
```shell
pip install -r requirements.txt
```
### Dataset

#### Training Dataset
KITTI VO dataset: contains 42104 images with a typical image resolution of 1241 × 376.

We sorted all 22 KITTI VO sequences and found 7 frames from sequence 17 and 20 having corresponding frames in the KITTI 2015 training set. Therefore we excluded these two sequences and used the remaining 20 stereo sequences as our training dataset.

#### Validation Dataset
We proveide the selected KITTI 141 dataset in [KITTI 141](https://drive.google.com/file/d/1lsuM3LUfwR2c_L1c1rDzJ_aG5bwtUdUk/view?usp=sharing).
It contains 141 frames from KITTI raw dataset that have corresponding frames in the KITTI 2015 dataset.

### Inference
Download the model-pth provided in [Best Model](https://drive.google.com/file/d/1NdEBdrUq8iM9ZkWjWmvfSph-3fPoE4yu/view?usp=sharing), and put them in `./checkpoint/`
```shell
sh inference.sh
```

