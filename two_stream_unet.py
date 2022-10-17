from distutils.util import change_root
from pyexpat import features
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from correlation_liteflownet import correlation
import getopt
import sys
import math
from KalmanFilter import KalmanModel, ExtendedKalmanModel, KalmanNet
from KalmanNetConv import *
from LatentRNN import LSTMModel, LSTMModelEval
import ipdb
# from LiteFlowNet import Network, backwarp #! dont call this copy the network
# torch.manual_seed(10)
# torch.manual_seed(42)
torch.manual_seed(101)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


##########################################################
arguments_strModel = 'default' # 'default', or 'kitti', or 'sintel'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
    # if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
    # if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end
##########################################################




backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow): #? just doing some sort of interpolation 
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return F.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin = 1) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x, y):
        '''
        input: x : BxCxHxW 
               y : Bx1 : labels
        output: scalar loss
        '''        
        batch_size = x.shape[0]
        x_flat = x.flatten(1,-1)        
        dist_mat = torch.cdist(x_flat, x_flat)**2
        # tt = torch.randint(0,1,(10,1))
        label_matrix = torch.logical_xor(y, y.permute(1,0)).long()
        loss = (1 - label_matrix)*dist_mat + label_matrix*torch.clamp(self.margin - dist_mat, max = 0.0)   
        return torch.sum(loss)/batch_size     


class FlowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, x):
                # x1 = self.netOne(x)
                # print("input shape: " , x.shape , " with input shape: ", x1.shape) 
                # print("increasing channels to input to 32 to match network")
                x_ = torch.cat([x[:,0:1,:,:]]*32, dim=1)  # netTwo requires 32 channels as input 
                x2 = self.netTwo(x_)
                x3 = self.netThr(x2)
                x4 = self.netFou(x3)
                x5 = self.netFiv(x4)
                x6 = self.netSix(x5)
                return [x, x2, x3, x4, x5 ]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackwarp = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel] #! yeh kaha se aae?

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )                

                if intLevel == 6:
                    self.netUpflow = None

                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)


                if intLevel >= 4:
                    self.netUpcorr = None

                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                )
            

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne) # only when intLevel = 2 = intLevel[0]
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo) # only when intLevel = 2 = intLevel[0]

                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)               

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)

                if self.netUpcorr is None:
                    tmp_ = correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1)
                    ipdb.set_trace()
                    tenCorrelation = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1), negative_slope=0.1, inplace=False)

                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=2), negative_slope=0.1, inplace=False))

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)


        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )


                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 130, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                )
            # end

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne) # only when intLevel = 2 = intLevel[0]
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo) # only when intLevel = 2 = intLevel[0]

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
            
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesOne, tenFeaturesTwo, tenFlow ], 1))


        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                    )

                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 3, 2, 2, 1, 1 ][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][intLevel]))
                    )

                self.netScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = (tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)).square().sum(1, True).sqrt().detach()

                tenDist = self.netDist(self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), self.netFeat(tenFeaturesOne) ], 1)))
                tenDist = tenDist.square().neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                tenDivisor = tenDist.sum(1, True).reciprocal()

                tenScaleX = self.netScaleX(tenDist * F.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * F.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

                return torch.cat([ tenScaleX, tenScaleY ], 1)

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        self.netSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        self.netRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-liteflownet/network-' + arguments_strModel + '.pytorch', file_name='liteflownet-' + arguments_strModel).items() })    

    def forward(self, image1, image2):
        #? subtract by 0.5 for US iamges ? 
        # image1[:, 0, :, :] = image1[:, 0, :, :] #- 0.411618
        # image1[:, 1, :, :] = image1[:, 1, :, :] #- 0.434631
        # image1[:, 2, :, :] = image1[:, 2, :, :] #- 0.454253

        # image2[:, 0, :, :] = image2[:, 0, :, :] #- 0.410782
        # image2[:, 1, :, :] = image2[:, 1, :, :] #- 0.433645
        # image2[:, 2, :, :] = image2[:, 2, :, :] #- 0.452793
        image1 = torch.cat([image1]*3, dim=1)
        image2 = torch.cat([image2]*3, dim=1)
        features1 = self.netFeatures(image1) # first netry is OG image 
        features2 = self.netFeatures(image2)

        image1_list = [image1] 
        image2_list = [image2] 

        # creating list of OG images with different sizes based of feature size
        for intLevel in [ 1, 2, 3, 4 ]:
            # print("intlevel = " , intLevel)
            image1_list.append(F.interpolate(input=image1_list[-1], size=(features1[intLevel].shape[2], features1[intLevel].shape[3]), mode='bilinear', align_corners=False))
            image2_list.append(F.interpolate(input=image2_list[-1], size=(features2[intLevel].shape[2], features2[intLevel].shape[3]), mode='bilinear', align_corners=False))    

        flow = None

        for intLevel in [-1, -2, -3, -4]: #[ -2, -3, -4, -5 ]:
            # print("intLevel = " , intLevel)
            intLevel_ = intLevel - 1
            flow = self.netMatching[intLevel_](image1_list[intLevel], image2_list[intLevel], features1[intLevel], features2[intLevel], flow)            
            flow = self.netSubpixel[intLevel_](image1_list[intLevel], image2_list[intLevel], features1[intLevel], features2[intLevel], flow)
            flow = self.netRegularization[intLevel_](image1_list[intLevel], image2_list[intLevel], features1[intLevel], features2[intLevel], flow)
            
        # return flow #* 20.0 
        return flow, None 

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, type='training'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),            
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5, inplace=False), #default inplace is also False            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5, inplace=False)
        )

    def forward(self, x):
        # ipdb.set_trace()
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5, inplace=False)
        )

    def forward(self, x):
        # ipdb.set_trace()
        return self.single_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_layers=1, mid_channels=None):
        super().__init__()
        if conv_layers == 1:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2), # kernel size is the input               
                SingleConv(in_channels, out_channels)                                
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2), # kernel size is the input               
                DoubleConv(in_channels, out_channels, (in_channels + out_channels)//2)
            )            

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, conv_layers=1, attention_flag=False):
        super().__init__()

        self.attention_flag = attention_flag
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if not attention_flag:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            else:
                pass

            if conv_layers == 1:
                self.conv = SingleConv(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if not attention_flag:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            else:
                pass

            if conv_layers == 1:
                self.conv = SingleConv(in_channels, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2, high_res_flag=False):

        if not self.attention_flag:

            if high_res_flag:
                x = torch.cat([x2, x1], dim=1)
            else:
                x1 = self.up(x1)
                # input is CHW
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                #! add non-zero padding 
                # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                #                 diffY // 2, diffY - diffY // 2])
                x1 = F.pad(x1, [diffX - diffX // 2, diffX // 2,
                                diffY - diffY // 2, diffY // 2])                            
                # if you have padding issues, see
                # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
                # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
                x = torch.cat([x2, x1], dim=1)

        else:
            x = x1
            
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, in_channels_up, in_channels_skip, device = 'cuda') -> None:
        super(AttentionGate, self).__init__()
        out_channels = in_channels_skip
        self.conv_skip = nn.Conv2d(in_channels_skip, out_channels,kernel_size=1, stride=2)
        self.conv_up = nn.Conv2d(in_channels_up, out_channels,kernel_size=1, stride=1)
        self.non_linearity = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(out_channels, 1, kernel_size= 1)
        self._sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_skip.to(device)
        self.conv_up.to(device)
        self.conv_out.to(device)

    def forward(self, _up, _skip):

        # map features to same channel dim         
        _skip_conv = self.conv_skip(_skip)
        _up_conv = self.conv_up(_up)
        # take summation 
        _out = _skip_conv + _up_conv
        # relu + map to single channel
        _out_conv = self.conv_out(self.non_linearity(_out))
        # sigmoid to scale between 0 - 1 
        _out_conv = self._sigmoid(_out_conv)
        # upsample 
        _out_conv = self.up(_out_conv)

        return _out_conv

class UNet(nn.Module):
    '''
    Based off https://arxiv.org/pdf/1505.04597.pdf

    the network expects an input of shape B x n_channels x H x W and outputs a segmentation mask without sigmoid  (sigmoid subsumed in BCEwithlogits) 
    
    n_channels depends on variant of input: 
    ***
    some examples: only image then n_channels = 1 
                   image and flow concatenated then n_channels = 2 
                   image and flow added then n_channels = 1 
                   pre-processing image and flow through conv2d, then n_channels = output of conv2d
    ***

    inputs: 
            input channels : n_channels
            # of classes to predict : n_classes 
            depth of network : n_depth
    input flags: 
            bilinear : whether to use bilinear interpolation during upsampling or not 
            conv_layers : how many conv2d to use in each downsampling and upsampling in encoder and decoder, current options: 1 or 2
            classification_flag : when true will also compute classification/contrastive loss by taking the last feature of encoder and passing to linear layer
            attention_flag : if true will use attention gates when upsampling in decoder 
    '''
    def __init__(self, args, **kwargs):

        super(UNet, self).__init__()
        self.batch_size = args.batch_size
        self.n_channels = kwargs['n_channels']
        self.n_classes = kwargs['n_classes']
        self.depth = kwargs['n_depth']
        self.bilinear = kwargs['bilinear']
        self.device = kwargs['device']

        self.classification_flag = args.classification_flag
        self.attention_flag = args.attention_flag
        self.multi_attn = args.multi_attn
        self.conv_layers = args.conv_layers
        self.kalman_flag = args.kalman_flag

        feature_size = 64
        features = [feature_size*(2**i) for i in range(self.depth)]
        self.features = features

        self.inc = DoubleConv(self.n_channels, features[0])
        self.down1 = Down(features[0], features[1], self.conv_layers)
        self.down2 = Down(features[1], features[2], self.conv_layers)
        self.down3 = Down(features[2], features[3], self.conv_layers)
        self.down4 = Down(features[3], features[4], self.conv_layers)#extra
        factor = 2 if self.bilinear else 1

        self.up1 = Up(features[4], features[3] // factor, self.bilinear, self.conv_layers)#extra
        self.up2 = Up(features[3], features[2] // factor, self.bilinear, self.conv_layers)
        self.up3 = Up(features[2], features[1] // factor, self.bilinear, self.conv_layers)
        self.up4 = Up(features[1], features[0], self.bilinear, self.conv_layers)

        #? add fc layers for classification or contrastive 
        if self.classification_flag:
            in_features_ = (1024//factor)*16*16
            self.fc = nn.Linear(in_features_, 1) # only 2 classes 
        
        if self.attention_flag:
            self.att10 = AttentionGate(features[4], features[3]) #extra
            self.att11 = AttentionGate(features[3], features[2])
            self.att12 = AttentionGate(features[2], features[1])
            self.att13 = AttentionGate(features[1], features[0])
            
            self.up1 = Up(features[3], features[3], self.bilinear, self.conv_layers, self.attention_flag)#extra
            self.up2 = Up(features[2], features[2], self.bilinear, self.conv_layers, self.attention_flag)
            self.up3 = Up(features[1], features[1], self.bilinear, self.conv_layers, self.attention_flag)
            self.up4 = Up(features[0], features[0], self.bilinear, self.conv_layers, self.attention_flag)

            if self.multi_attn > 0:
                self.att21 = AttentionGate(features[3], features[2])
                self.att22 = AttentionGate(features[2], features[1])
                self.att23 = AttentionGate(features[1], features[0])
                
                self.att31 = AttentionGate(features[3], features[2])
                self.att32 = AttentionGate(features[2], features[1])
                self.att33 = AttentionGate(features[1], features[0])

                self.att41 = AttentionGate(features[3], features[2])
                self.att42 = AttentionGate(features[2], features[1])
                self.att43 = AttentionGate(features[1], features[0])                                            

        self.outc = OutConv(features[0], self.n_classes)

    def forward(self, x, batch_size=10, new_video_flag=False, gauss_flag=False):
        '''
            x can contain multiple inputs from previous time-steps concatenated along batch so that 
            feature extraction can be done in parallel 
        '''
        # prev_img = x[3*batch_size:4*batch_size].clone()        
        _x = x[0:batch_size]
        # import ipdb; ipdb.set_trace()
        x1 = self.inc(_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   
        x5 = self.down4(x4)

        if self.attention_flag:
            # this will work for single image, need to change for multiple images as first attention mask would be different
            if self.multi_attn > 0:                
                _scale = self.multi_attn+1

                #* downsample flow to different size 
                _flow = x[batch_size:] #torch.flip(x[batch_size:], dim=0) #
                _flow_l1 = torch.cat([F.interpolate(_flow, scale_factor= 1)]*self.features[0], dim=1)
                _flow_l2 = torch.cat([F.interpolate(_flow, scale_factor = 0.5)]*self.features[1], dim=1)
                _flow_l3 = torch.cat([F.interpolate(_flow, scale_factor = 0.25)]*self.features[2], dim=1)
                # find attention with own feature at different level
                att_other_list = []
                att_self1 = self.att11(x4, x3[0:batch_size])
                att_other21 = self.att21(x4, _flow_l3[0:batch_size])
                att_other31 = self.att31(x4, _flow_l3[batch_size:2*batch_size])
                att_other41 = self.att41(x4, _flow_l3[2*batch_size:3*batch_size])
                # att1 = torch.cat([att_self1, att_other21, att_other31, att_other41], dim=1)
                att1 = (att_self1 + att_other21 + att_other31 + att_other41)/(self.multi_attn + 1)
                #* 0:batch_size is current image 
                #* 3*batch_size: 4*batch_size is {current - 1} image 
                #* 2*batch_size: 3*batch_size is {current - 2} image  and so on 
                y = x3*att1
                # y = att1[:,0:1]*x3[0:batch_size] + att1[:,3:4]*x3[3*batch_size:4*batch_size] + att1[:,2:3]*x3[2*batch_size:3*batch_size] + att1[:,1:2]*x3[batch_size:2*batch_size]
                
                # att1 = F.softmax(torch.cat([att_self1, att_other1],dim=1), dim=1)                
                # att1 = torch.cat([att_self1, att_other1],dim=1)
                # y = att1[:,0:1] * x3[0:batch_size] + att1[:,0:1] * att1[:,1:2] * x3[batch_size:2*batch_size]                
                y = self.up2(y, None)
                # find attention between y and previous images 
                att_self2 = self.att12(y, x2[0:batch_size])
                att_other22 = self.att22(y, _flow_l2[0:batch_size])
                #! add stuff here
                att_other32 = self.att32(y, _flow_l2[batch_size:2*batch_size])
                att_other42 = self.att42(y, _flow_l2[2*batch_size:3*batch_size])
                # att2 = torch.cat([att_self2, att_other22, att_other32, att_other42],dim=1)
                att2 = (att_self2 + att_other22 + att_other32 + att_other42)/(self.multi_attn + 1)
                # y = att2[:,0:1]*(x2[0:batch_size] + att2[:,3:4]*(x2[3*batch_size:4*batch_size] + att2[:,2:3]*(x2[2*batch_size:3*batch_size] + att2[:,1:2]*x2[batch_size:2*batch_size] )))
                y = att2 * x2
                # att2 = torch.cat([att_self2, att_other2], dim=1)
                # y = att2[:,0:1] * x2[0:batch_size] + att2[:,0:1] * att2[:,1:2] * x2[batch_size:2*batch_size]
                y = self.up3(y, None)

                att_self3 = self.att13(y, x1[0:batch_size])
                att_other23 = self.att23(y, _flow_l1[0:batch_size])
                #! add stuff here
                att_other33 = self.att33(y, _flow_l1[batch_size:2*batch_size])
                att_other43 = self.att43(y, _flow_l1[2*batch_size:3*batch_size])                

                att3 = (att_self3 + att_other23 + att_other33 + att_other43)/(self.multi_attn + 1)
                y = att3 * x1
                # y = att3[:,0:1]*(x1[0:batch_size] + att3[:,3:4]*(x1[3*batch_size:4*batch_size] + att3[:,2:3]*(x1[2*batch_size:3*batch_size] + att3[:,1:2]*x1[batch_size:2*batch_size])))
                
                # att3 = torch.cat([att_self3, att_other3], dim=1)
                # y = att3[:,0:1] * x1[0:batch_size] + att3[:,0:1] * att3[:,1:2] * x1[batch_size:2*batch_size]
                y = self.up4(y, None)
                
            else:
                att_self0 = self.att10(x5, x4)#extra
                y = self.up1(att_self0 * x4, None)#extra

                att_self1 = self.att11(y, x3)#replace y with x4
                y = self.up2(att_self1 * x3, None)

                att_self2 = self.att12(y, x2)
                y = self.up3(att_self2 * x2, None)

                att_self3 = self.att13(y, x1)
                y = self.up4(att_self3 * x1, None)

        else:
            y = self.up1(x5, x4)
            y = self.up2(y, x3)        
            y = self.up3(y, x2)
            y = self.up4(y, x1)

        # for plotting 
        # last_encoder_feature = nn.Upsample(size=(64,64), mode='bilinear')(x4)

        logits = self.outc(y)

        return logits, None, None


class UNetBKF(nn.Module):
    '''
        UNet for ConvKalman 
    '''
    def __init__(self, args, **kwargs):
        super(UNetBKF, self).__init__()
        self.batch_size = args.batch_size
        self.n_channels = kwargs['n_channels']
        self.n_classes = kwargs['n_classes']
        self.depth = kwargs['n_depth']
        self.bilinear = kwargs['bilinear']
        self.device = kwargs['device']
        self.kalman_flag = args.kalman_flag
        self.classification_flag = args.classification_flag
        self.attention_flag = args.attention_flag
        self.multi_attn = args.multi_attn
        self.conv_layers = args.conv_layers
        self.gauss_flag = args.gauss_flag
        self.transformer_flag = args.transformer_flag
        self.high_res_flag = args.high_res_flag


        feature_size = kwargs['unet_channel_start']
        features = [feature_size*(2**i) for i in range(self.depth)] 
        self.features = features
        #* down sample (encoder)
        self.inc = DoubleConv(self.n_channels, features[0])
        self.down1 = Down(features[0], features[1], self.conv_layers)
        self.down2 = Down(features[1], features[2], self.conv_layers)
        self.down3 = Down(features[2], features[3], self.conv_layers)
        # self.down4 = Down(features[3], features[4] // factor, self.conv_layers)
        
        factor = 2 if self.bilinear else 1

        # self.up1 = Up(features[4], features[3] // factor, self.bilinear, self.conv_layers)
        self.up2 = Up(features[3], features[2] // factor, self.bilinear, self.conv_layers)
        self.up3 = Up(features[2], features[1] // factor, self.bilinear, self.conv_layers)
        self.up4 = Up(features[1], features[0], self.bilinear, self.conv_layers)

        #? final layer 
        self.outc = OutConv(features[0], self.n_classes)        
        if args.eval == True:
            if self.gauss_flag:
                self.kalman_model = KalmanNetConvGaussVal(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], kwargs['width'], embed_channels=kwargs['kf_channels'])
            else:
                self.kalman_model = KalmanNetConvVal(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], kwargs['width'], embed_channels=kwargs['kf_channels'])
                # self.kalman_model = LSTMModelEval(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], kwargs['width'])
            print("Running Evaluation Model")
        else:
            if self.gauss_flag:
                self.kalman_model = KalmanNetConvGauss(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], kwargs['width'], embed_channels=kwargs['kf_channels'])
            else:
                self.kalman_model = KalmanNetConv(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], \
                                                kwargs['width'], embed_channels=kwargs['kf_channels'], transformer_flag=self.transformer_flag, high_res_flag = self.high_res_flag)
                # self.kalman_model = LSTMModel(self.device, self.batch_size, kwargs['in_channels'], kwargs['height'], kwargs['width'])


    def forward(self, x, new_video_flag=False, gauss_flag=False, batch_size=10):
        '''
            x can contain multiple inputs from previous time-steps concatenated along batch so that 
            feature extraction can be done in parallel 
            This needs the whole trajectory to be passed to the model
            Dims: B x T x C x H x W
        '''
        # reshape input to B*T x C x H x W 
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   

        BT, C_, H_, W_ = x4.shape
        
        x4 = x4.view(B, T, C_, H_, W_)

        if self.gauss_flag:
            #* original + gauss 
            x4, x4_sigma = self.kalman_model(x4, new_video_flag=new_video_flag)
            x4 = x4.view(B*T, C_, H_, W_)
            x4_sigma = x4_sigma.view(B*T, C_, H_, W_)

            # concat x4, x4_sigma along batch as 
            x4_new = torch.cat([x4, x4_sigma], dim=0)

            y = self.up2(x4_new, torch.cat([x3, x3], dim=0))
            y = self.up3(y, torch.cat([x2, x2], dim=0))
            y = self.up4(y, torch.cat([x1, x1], dim=0))

            # y_mean, y_logstd = y[:B*T], y[B*T,:]
            # y_std = torch.exp(y_logstd)
            # y_sample = y_mean + y_std*torch.randn(B*T, C_, H_, W_)

            # find logits using y_sample
            logits = self.outc(y)
            logits_mean, logits_logstd = logits[:B*T], logits[B*T:]
            logits_std = torch.exp(logits_logstd)
            logits_std = logits_std.view(B, T, C, H, W)
            logits_mean = logits_mean.view(B, T, C, H, W)
            logits_sample = logits_mean + logits_std*torch.randn(B, T, C, H, W, device = self.device)            
            
            return logits_sample, logits_mean, logits_std

        else:
            #* original 
            x4 = self.kalman_model(x4, new_video_flag=new_video_flag)
            
            if self.high_res_flag:
                x4 = x4.view(B*T, C_//2, 2*H_, 2*W_)
            else:
                x4 = x4.view(B*T, C_, H_, W_)

            y = self.up2(x4, x3, high_res_flag=self.high_res_flag)
            y = self.up3(y, x2)
            y = self.up4(y, x1)

            # for plotting 
            last_encoder_feature = None #nn.Upsample(size=(64,64), mode='bilinear')(x4)

            logits = self.outc(y)
            logits = logits.view(B, T, C, H, W)

            return logits, None, None

    def forward_process_model(self, x, batch_size = 10, new_video_flag=False, gauss_flag=False):
        '''
        input: single image frame 
        output: generate new state by forward propagation through process model only
        output propagated through decoder block
        '''
        # reshape input to B*T x C x H x W 
        B, T, C, H, W = x.shape #T = 1
        x = x.view(-1, C, H, W)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   
        
        BT, C_, H_, W_ = x4.shape
        
        #x4_list seq_len x 1 x C_ x H_ x W_
        x4_list = self.kalman_model.process_model(x4, new_video_flag=new_video_flag)
    
        # x4 = x4_list.view(B*T, C_, H_, W_)
        B_ = x4_list.shape[0]

        y = self.up2(x4_list, torch.cat([x3]*B_, dim=0))        
        y = self.up3(y, torch.cat([x2]*B_, dim=0))
        y = self.up4(y, torch.cat([x1]*B_, dim=0))

        logits = self.outc(y)
        _, C_, H_, W_ = logits.shape
        logits = logits.view(B_, 1, C_, H_, W_)

        return logits, None, None

class TwoStreamUNet(nn.Module):
    '''
    main class for needle tracking using unet as a base and adding different ideas 
    we use <var>_flag to decide which particular idea to try 
    <var>_flag are store_action arg-pasrsers 
    For examples if using pure_images, 
                 or using additional classification_loss 
                 or using attention gates 
                 or using learned optical flow 
    '''
    # def __init__(self, batch_size, spatial_in_channel, temporal_in_channel, out_channel, n_classes, \
    #             pure_images_flag=False, conv_layers=1, learned_flow=False, classification_flag=False,\
    #                 attention_flag = False, multi_attn = 0, device = 'cuda', motion_flag=False, kalman_flag=False) -> None:
    def __init__(self,args,**kwargs) -> None:                    

        super(TwoStreamUNet, self).__init__()
        assert args is not None, 'args cannot be None'
        
        self.pure_images_flag = args.pure_images
        self.temporal_in_channel = args.n_flow
        self.learned_flow = args.learned_flow
        self.multi_attn = args.multi_attn
        self.batch_size = args.batch_size
        self.kalman_flag = args.kalman_flag
        self.flow_flag = args.flow_flag
        self.process_model_flag = args.process_model_flag
        
        self.n_classes = kwargs['n_classes']
        self.spatial_in_channel = kwargs['spatial_in_channel']
        self.out_channel = kwargs['out_channels']


        if self.learned_flow:
            self.flownet = FlowNet()
            temporal_in_channel = 2
            assert temporal_in_channel == 2, "flownet outputs 2 channels for x,y, can't normalize this"
            self.temporal_conv = DoubleConv(2, self.out_channel)

        if self.pure_images_flag:
            self.spatial_conv = DoubleConv(self.spatial_in_channel, self.out_channel) 
            self.n_channels = self.out_channel 
        elif self.flow_flag:
            self.spatial_conv = DoubleConv(self.spatial_in_channel, self.out_channel) 
            self.temporal_conv = DoubleConv(self.temporal_in_channel, self.out_channel) 
            self.n_channels = self.out_channel
        else:
            if args.n_flow > 1:
                self.n_channels = 1 + args.n_flow
            else:
                self.n_channels = 1 #out_channel # 16 
        
        
        if self.kalman_flag:
            in_channels =kwargs['unet_channel_start']*2**3 #256 #512 # channels of last encoded state UNet
            height = 32 #this is 256/(2**3) height of last encoded state UNet
            width = 32 # width of last encoded state UNet
            B = self.batch_size
            self.UNet = UNetBKF(args,n_channels=self.n_channels, in_channels=in_channels, height=height, width=width,**kwargs)
        else:
            self.UNet = UNet(args, n_channels=self.n_channels, **kwargs)

    def forward(self, image, flow=None, image_prev=None, new_video_flag = False):
        '''
            image is the image at last time step (t)
            image_prev are images from (t-k) time steps: t-1, t-2, .... 
        '''
        
        if self.pure_images_flag:
            x_spatial = self.spatial_conv(image)
            logits = self.UNet(x_spatial)
            x_temporal = None
        
        elif self.learned_flow:
            #? no need to resize images to 448, 1025; use the 256 size and
            #? start from second or third feature layer
            #? no need to use last layer of M, S and R as well
            img_H, img_W = image.shape[-2], image.shape[-1]
            
            w_32 = int(math.floor(math.ceil(img_H / 32.0) * 32.0))
            h_32 = int(math.floor(math.ceil(img_W / 32.0) * 32.0))

            image1_resized = F.interpolate(input=image_prev, size=(h_32, w_32), mode='bilinear', align_corners=False)
            image2_resized = F.interpolate(input=image, size=(h_32, w_32), mode='bilinear', align_corners=False)

            flow, flow_tmp = self.flownet(image1_resized, image2_resized)
            flow_resized = F.interpolate(input=flow, size=(img_H, img_W), mode='bilinear', align_corners=False)

            x_spatial = self.spatial_conv(image)
            x_temporal = self.temporal_conv(flow_resized) 

        elif self.flow_flag:
            batch_size = image.shape[0]            
            x_spatial = self.spatial_conv(image.type(torch.float32))
            x_temporal = self.temporal_conv(flow)
            image = x_spatial + x_temporal #torch.cat([x_spatial, x_temporal], dim=1)

        else:
            # kalman_flag, attention_flag, 
            batch_size = image.shape[0]
            flow=None
            x_spatial = torch.zeros(10,32,128,128)
            x_temporal = torch.zeros(10,32,128,128)

            if self.multi_attn > 0:
                channels = image_prev.shape[1] #  should be same as multi_attn
                h, w = image_prev.shape[-2], image_prev.shape[-1]
                _image_prev = image_prev.view(batch_size*channels, 1, h, w).type(torch.float32)
                flow = flow.view(batch_size*channels, 1, h, w).type(torch.float32)
                image = torch.cat([image.type(torch.float32), flow], dim=0) # concatenate along batch 

            elif image_prev is not None:
                image = torch.cat([image, image_prev], dim=-3)
            else:
                image = image
        
        if self.kalman_flag and self.process_model_flag:
            logits, mean, sigma = self.UNet.forward_process_model(image, batch_size=batch_size, new_video_flag=new_video_flag)
        else:
            logits, mean, sigma = self.UNet(image, batch_size=batch_size, new_video_flag=new_video_flag)
        
        return logits, mean, sigma, flow
        

# class TwoStreamUNetLateFusion(nn.Module):
#     '''
#     ignor this class for time being 
#     '''
#     def __init__(self,spatial_in_channel, temporal_in_channel, out_channel, n_classes, pure_images_flag=False, conv_layers=1, bilinear=False) -> None:
#         # super(TwoStreamUNet, self).__init__(spatial_in_channel, temporal_in_channel, out_channel, n_classes, pure_images_flag, conv_layers)
#         super(TwoStreamUNetLateFusion, self).__init__()

#         # ipdb.set_trace()        
#         self.temporal_in_channel = temporal_in_channel
#         self.n_classes = n_classes
#         self.n_channels = 1
#         self.bilinear = bilinear 

#         # encoder set for spatial 
#         # self.UNet = UNet(self.n_channels, n_classes, conv_layers)
#         encoder_scale = 2
#         self.inc_s = DoubleConv(spatial_in_channel, 64//encoder_scale)
#         self.down1_s = Down(64//encoder_scale, 128//encoder_scale, conv_layers)
#         self.down2_s = Down(128//encoder_scale, 256//encoder_scale, conv_layers)
#         self.down3_s = Down(256//encoder_scale, 512//encoder_scale, conv_layers)
#         factor = 2 if self.bilinear else 1
#         # self.down4_s = Down(512, 1024 // factor, conv_layers)
#         self.down4_s = Down(512//encoder_scale, 1024//encoder_scale, conv_layers)

#         # encoder set for temporal 
#         self.inc_t = DoubleConv(temporal_in_channel, 64//encoder_scale)
#         self.down1_t = Down(64//encoder_scale, 128//encoder_scale, conv_layers)
#         self.down2_t = Down(128//encoder_scale, 256//encoder_scale, conv_layers)
#         self.down3_t = Down(256//encoder_scale, 512//encoder_scale, conv_layers)
#         factor = 2 if self.bilinear else 1
#         # self.down4_t = Down(512, 1024 // factor, conv_layers)
#         self.down4_t = Down(512//encoder_scale, 1024//encoder_scale, conv_layers)

#         # common decoder concatenate both encoder outputs | spatial, temporal or both        
#         scale = 1
#         self.up1 = Up(scale*1024, scale*512 // factor, bilinear, conv_layers)
#         self.up2 = Up(scale*512, scale*256 // factor, bilinear, conv_layers)
#         self.up3 = Up(scale*256, scale*128 // factor, bilinear, conv_layers)
#         self.up4 = Up(scale*128, 64, bilinear, conv_layers)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, image, flow):
#         # find features for spatial input : image 
#         x1_s = self.inc_s(image)
#         x2_s = self.down1_s(x1_s)
#         x3_s = self.down2_s(x2_s)
#         x4_s = self.down3_s(x3_s)
#         x5_s = self.down4_s(x4_s)

#         # find features for temporal input : flow 
#         x1_t = self.inc_t(flow)
#         x2_t = self.down1_t(x1_t)
#         x3_t = self.down2_t(x2_t)
#         x4_t = self.down3_t(x3_t)
#         x5_t = self.down4_t(x4_t)

#         # concatenate x5_s, x5_t
#         x5 = torch.cat([x5_s, x5_t], dim = 1)
#         x4 = torch.cat([x4_s, x4_t], dim = 1)
#         x3 = torch.cat([x3_s, x3_t], dim = 1)
#         x2 = torch.cat([x2_s, x2_t], dim = 1)
#         x1 = torch.cat([x1_s, x1_t], dim = 1)

#         # element wise mulitply 
#         # x5 = x5_s * x5_t        
#         # x4 = x4_s * x4_t        
#         # x3 = x3_s * x3_t        
#         # x2 = x2_s * x2_t        
#         # x1 = x1_s * x1_t        

#         y = self.up1(x5, x4)
#         # print("y shape: " , y.shape)
#         y = self.up2(y, x3)
#         # print("y shape: " , y.shape)
#         y = self.up3(y, x2)
#         # print("y shape: " , y.shape)
#         y = self.up4(y, x1)
#         # print("y shape: " , y.shape)

#         # ipdb.set_trace()
#         logits = self.outc(y)        
        
#         return logits, None, None         


## add late fusion instead of early fusion 
## add multiple fusions instead of single fusion 
## reduce network size 
## add correlation thingy 


