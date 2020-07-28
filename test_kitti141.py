from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import numpy as np
import time
import math

from dataloader import dataloader_kitti141 as DL
from models import *
from path import Path

parser = argparse.ArgumentParser(description='LidarStereoNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='kitti',
                    help='datapath')
parser.add_argument('--datapath', default='./data',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./results/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default="1", type=str,
                    help='Which GPU to use? (default:3)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
save_path = Path(args.savemodel)
save_path.makedirs_p()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


left_test, right_test, displ_test, dispr_test = DL.test_Kitti141(args.datapath)
test_dataset  = DL.TestLoader(False, left_test,  right_test, displ_test,  dispr_test )
TestImgLoader = DataLoader(dataset=test_dataset,  batch_size=1, num_workers=1, shuffle=False, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


if os.path.isfile(args.loadmodel):
    print("=> loading checkpoint '{}'".format(args.loadmodel))
    checkpoint = torch.load(args.loadmodel)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
else:
    print("=> no checkpoint found at '{}'".format(args.loadmodel))  


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
   
def test(imgL, imgR, disp_L, disp_R):

    model.eval()
    # Pad images such that the dimensions are multiples of 32
    shape = imgL.shape
    height_new = int(np.ceil(shape[2]/32.)*32)
    width_new  = int(np.ceil(shape[3]/32.)*32)

    padding = (0, width_new-shape[3], 0, height_new-shape[2])
    imgL = F.pad(imgL, padding, "constant", 0)
    imgR = F.pad(imgR, padding, "constant", 0)
    disp_L = F.pad(disp_L, padding, "constant", 0)
    disp_R = F.pad(disp_R, padding, "constant", 0)

    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))
    disp_L   = Variable(torch.FloatTensor(disp_L))
    disp_R   = Variable(torch.FloatTensor(disp_R))
    if args.cuda:
        imgL, imgR, disp_L, disp_R = imgL.cuda(), imgR.cuda(), disp_L.cuda(), disp_R.cuda()

    with torch.no_grad():
        outputl3, outputr3 = model(imgL,imgR, disp_L, disp_R)

    # Change output to original shape
    outputl3  = outputl3[:, :shape[2], :shape[3]]
    outputr3  = outputr3[:, :shape[2], :shape[3]]

    pred_disp = outputl3.data.cpu()
    output = torch.squeeze(pred_disp,1)[:,:,:]

    torch.cuda.empty_cache()
    return output


def main():   

    # TEST ##
    num_samples = len(left_test)

    for batch_idx, (imgL, imgR, dispL, dispR) in enumerate(TestImgLoader):
        output = test(imgL, imgR, dispL, dispR)

        # save prediction
        disp_est = torch.squeeze(output).numpy()
        skimage.io.imsave(args.savemodel+'images/'+ left_test[batch_idx][-13:], (disp_est*256).astype('uint16'))
        print('Inference Image: %s' %(left_test[batch_idx][-13:]))


if __name__ == '__main__':
   main()
