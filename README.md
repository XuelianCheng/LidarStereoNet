## LidarStereoNet
This is the code for our CVPR 2019 paper `Noise-Aware Unsupervised Deep Lidar-Stereo Fusion`

[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Noise-Aware_Unsupervised_Deep_Lidar-Stereo_Fusion_CVPR_2019_paper.pdf)
[arxiv](https://arxiv.org/abs/1904.03868)
[video](https://youtu.be/8wBzUY8bAvU)

The implementation of our model part is heavily borrow from [PSMNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf) and [Sparsity Invariant CNNs](http://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf). 

### Citing
If you find this code useful, please consider to cite our work.

```
@inproceedings{cheng2019cvpr,
title={Noise-Aware Unsupervised Deep Lidar-Stereo Fusion},
author={Cheng, Xuelian and Zhong, Yiran and Dai, Yuchao and Ji, Pan and Li, Hongdong},
booktitle={Proc. IEEE Conf. Comp. Vis. Patt. Recogn.},
year={2019}
}
```

## Installation

### Environment

1. Python 2.7.*
2. CUDA 10.0
3. PyTorch 1.0.0
4. TorchVision 0.4.0

### Install
Create a  virtual environment and activate it.
```shell
pip install -r requirements.txt
```

If the requirements.txt doesn't work, please try the following commands directly:
```shell
pip install torch==1.0.0 torchvision==0.4.0
pip install scikit-image path.py
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

To evaluate our results on the selected KITTI 141 dataset , please directly use this link [Inference Images](https://drive.google.com/file/d/1XnrEU6Xwsok20EdFoSswkmkgdx1dHNUy/view?usp=sharing) and use matlab code to compute.

```shell
run Inference_kitti141.m
```

