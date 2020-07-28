import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import utils.readpfm as rp
import utils.preprocess as preprocess
import glob
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader_png(path):
    if os.path.exists(path):
        disp= Image.open(path)
    else:
        disp = None
    return disp

def disparity_loader(path):
    if os.path.exists(path):
        disp, scaleL = rp.readPFM(path)
    else:
        disp = None
        scaleL = None
    return disp, scaleL


def test_Kitti141(filepath):
    left  = glob.glob(os.path.join(filepath, 'Kitti_141', 'image_02', '*.png'))
    right = glob.glob(os.path.join(filepath, 'Kitti_141', 'image_03', '*.png'))
    displ = glob.glob(os.path.join(filepath, 'Kitti_141', 'velodyne_points', '*.png'))
    dispr = glob.glob(os.path.join(filepath, 'Kitti_141', 'velodyne_points_right', '*.png'))     
    return sorted(left), sorted(right), sorted(displ), sorted(dispr)  

class TestLoader(Dataset):
    def __init__(self, training, left, right, left_disparity, right_disparity, loader=default_loader, dploader=disparity_loader_png):
        self.left   = left
        self.right  = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity

        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left   = self.left[index]
        right  = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        left_img  = self.loader(left)
        right_img = self.loader(right)
        
        w, h  = left_img.size
        dataL = self.dploader(disp_L)   
        dataR = self.dploader(disp_R) 
        
        if dataL is None:
            dataL = np.ones((h,w))*256
            dataR = np.ones((h,w))*256           
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)
        dataL = dataL/256
        dataR = dataR/256
            
        processed = preprocess.get_transform(augment=False)
        left_img  = processed(left_img)
        right_img = processed(right_img)
        dataL     = torch.unsqueeze(torch.from_numpy(dataL), 0)
        dataR     = torch.unsqueeze(torch.from_numpy(dataR), 0)
        return left_img, right_img, dataL, dataR
        
    def __len__(self):
        return len(self.left)
