from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .SparseConvNet import *

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()
        self.feature_disp_pre   = SparseConvNet()
        
        self.dres0 = nn.Sequential(convbn_3d(96, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right, displ, dispr):
        refimg_fea      = self.feature_extraction(left)
        targetimg_fea   = self.feature_extraction(right)
        reflidar_fea    = self.feature_disp_pre(displ) 
        targetlidar_fea = self.feature_disp_pre(dispr) 
        #reflidar_fea    = F.interpolate(reflidar_fea, [refimg_fea.size()[2],refimg_fea.size()[3]])
        #targetlidar_fea = F.interpolate(targetlidar_fea, [refimg_fea.size()[2],refimg_fea.size()[3]])

        ref_fea    = torch.cat((refimg_fea,reflidar_fea), 1)
        target_fea = torch.cat((targetimg_fea,targetlidar_fea), 1)

        #matching
        costl = Variable(torch.FloatTensor(ref_fea.size()[0], ref_fea.size()[1]*2, int(self.maxdisp/4),  ref_fea.size()[2],  ref_fea.size()[3]).zero_()).cuda()
        costr = Variable(torch.FloatTensor(target_fea.size()[0], target_fea.size()[1]*2, int(self.maxdisp/4),  target_fea.size()[2],  target_fea.size()[3]).zero_()).cuda()

        for i in range(int(self.maxdisp/4)):
            if i > 0 :
             costl[:, :ref_fea.size()[1], i, :,i:] = ref_fea[:,:,:,i:]
             costl[:, ref_fea.size()[1]:, i, :,i:] = target_fea[:,:,:,:-i]
            else:
             costl[:, :ref_fea.size()[1], i, :,:]  = ref_fea
             costl[:, ref_fea.size()[1]:, i, :,:]  = target_fea
        costl = costl.contiguous()
        
        for i in range(int(self.maxdisp/4)):
            if i > 0 :
             costr[:, :target_fea.size()[1], i, :,:(target_fea.size()[3]-i)] = target_fea[:,:,:,:-i]
             costr[:, target_fea.size()[1]:, i, :,:(target_fea.size()[3]-i)] = ref_fea[:,:,:,i:]
            else:
             costr[:, :target_fea.size()[1], i, :,:target_fea.size()[3]]   = target_fea
             costr[:, target_fea.size()[1]:, i, :,:target_fea.size()[3]]   = ref_fea
        costr = costr.contiguous()
        ########################
        costl0 = self.dres0(costl)
        costl0 = self.dres1(costl0) + costl0

        outl1, prel1, postl1 = self.dres2(costl0, None, None) 
        outl1 = outl1+costl0

        outl2, prel2, postl2 = self.dres3(outl1, prel1, postl1) 
        outl2 = outl2+costl0

        outl3, prel3, postl3 = self.dres4(outl2, prel1, postl2) 
        outl3 = outl3+costl0

        costl1 = self.classif1(outl1)
        costl2 = self.classif2(outl2) + costl1
        costl3 = self.classif3(outl3) + costl2
        ########################

        costr0 = self.dres0(costr)
        costr0 = self.dres1(costr0) + costr0

        outr1, prer1, postr1 = self.dres2(costr0, None, None) 
        outr1 = outr1+costr0

        outr2, prer2, postr2 = self.dres3(outr1, prer1, postr1) 
        outr2 = outr2+costr0

        outr3, prer3, postr3 = self.dres4(outr2, prer1, postr2) 
        outr3 = outr3+costr0

        costr1 = self.classif1(outr1)
        costr2 = self.classif2(outr2) + costr1
        costr3 = self.classif3(outr3) + costr2

        if self.training:
            costl1 = F.interpolate(costl1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)
            costl2 = F.interpolate(costl2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)

            costl1 = torch.squeeze(costl1,1)
            predl1 = F.softmax(costl1,dim=1)
            predl1 = disparityregression(self.maxdisp)(predl1)

            costl2 = torch.squeeze(costl2,1)
            predl2 = F.softmax(costl2,dim=1)
            predl2 = disparityregression(self.maxdisp)(predl2)

            ########################            
            costr1 = F.interpolate(costr1, [self.maxdisp,right.size()[2],right.size()[3]], mode='trilinear',align_corners=True)
            costr2 = F.interpolate(costr2, [self.maxdisp,right.size()[2],right.size()[3]], mode='trilinear',align_corners=True)

            costr1 = torch.squeeze(costr1,1)
            predr1 = F.softmax(costr1,dim=1)
            predr1 = disparityregression(self.maxdisp)(predr1)

            costr2 = torch.squeeze(costr2,1)
            predr2 = F.softmax(costr2,dim=1)
            predr2 = disparityregression(self.maxdisp)(predr2)

        ########################    
        costl3 = F.interpolate(costl3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)
        costl3 = torch.squeeze(costl3,1)
        predl3 = F.softmax(costl3,dim=1)
        predl3 = disparityregression(self.maxdisp)(predl3)
        
        costr3 = F.interpolate(costr3, [self.maxdisp,left.size()[2],right.size()[3]], mode='trilinear',align_corners=True)
        costr3 = torch.squeeze(costr3,1)
        predr3 = F.softmax(costr3,dim=1)
        predr3 = disparityregression(self.maxdisp)(predr3)

        if self.training:
            return predl1, predl2, predl3, predr1, predr2, predr3
        else:
            return predl3, predr3
