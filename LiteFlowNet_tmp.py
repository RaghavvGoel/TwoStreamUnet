#!/usr/bin/env python

from configparser import Interpolation
import getopt
import math
from matplotlib.pyplot import plot
import numpy
import PIL
import PIL.Image
import sys
import torch
import ipdb
import os
import cv2 
import numpy as np
import torch.nn.functional as F
# from utils_two_stream_unet import get_image

IMAGE_LOC  = 'JPEGImages'
MASK_LOC = 'SegmentationClass'

PARENT_FOLDER_TRAIN = 'data' #os.path.join(ROOT_FOLDER,'data')
PARENT_FOLDER_TEST = 'data/test' #os.path.join(ROOT_FOLDER, 'data/test')

LIST_OF_DATASETS_TRAIN = ['task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1']
# LIST_OF_DATASETS_TRAIN = ['task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1',
#                     'task_positives_153-2022_04_12_18_32_22-segmentation mask 1.1',
#                     'task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1',
#                     'task_positives_193-2022_04_18_19_26_03-segmentation mask 1.1',
#                     'task_positives_67-2022_04_18_19_11_58-segmentation mask 1.1',
#                     'task_positives_222-2022_04_19_22_31_15-segmentation mask 1.1',
#                     'task_positives_230-2022_04_12_18_39_39-segmentation mask 1.1'
#                     ]

LIST_OF_DATASETS_TEST = ['task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1',
                    ]


try:
    from correlation_liteflownet import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation_liteflownet'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default' # 'default', or 'kitti', or 'sintel'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strOut = './out.flo'

# ipdb.set_trace()
for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
    if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
    if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

##########################################################

class Network(torch.nn.Module):
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
            # end

            def forward(self, x):
                x1 = self.netOne(x)
                print("input shape: " , x.shape , " with input shape: ", x1.shape) 
                print("increasing channels to input to 32 to match network")
                x_ = torch.cat([x[:,0:1,:,:]]*32, dim=1)  # netTwo requires 32 channels as input 
                x2 = self.netTwo(x_)
                x3 = self.netThr(x2)
                x4 = self.netFou(x3)
                x5 = self.netFiv(x4)
                x6 = self.netSix(x5)
                return [x, x2, x3, x4, x5 , x6]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.fltBackwarp = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                if intLevel == 6:
                    self.netUpflow = None

                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

                # end

                if intLevel >= 4:
                    self.netUpcorr = None

                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)

                # end

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                )
            # end

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)

                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                # end

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)
                # end

                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1), negative_slope=0.1, inplace=False)

                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=2), negative_slope=0.1, inplace=False))

                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)
            # end
        # end

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

                # end

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
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)

                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesOne, tenFeaturesTwo, tenFlow ], 1))
            # end
        # end

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

                # end

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

                # end

                self.netScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
            # eny

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = (tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)).square().sum(1, True).sqrt().detach()

                tenDist = self.netDist(self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), self.netFeat(tenFeaturesOne) ], 1)))
                tenDist = tenDist.square().neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                tenDivisor = tenDist.sum(1, True).reciprocal()

                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

                return torch.cat([ tenScaleX, tenScaleY ], 1)
      
        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        # self.netMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5 ] ])
        self.netSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        # self.netSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5 ] ])
        self.netRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        # self.netRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5 ] ])

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-liteflownet/network-' + arguments_strModel + '.pytorch', file_name='liteflownet-' + arguments_strModel).items() })


    def forward(self, image1, image2, ultrasound_flag = False):
        #? subtract by 0.5 for US iamges ? 
        if not ultrasound_flag:
            image1[:, 0, :, :] = image1[:, 0, :, :] - 0.411618
            image1[:, 1, :, :] = image1[:, 1, :, :] - 0.434631
            image1[:, 2, :, :] = image1[:, 2, :, :] - 0.454253

            image2[:, 0, :, :] = image2[:, 0, :, :] - 0.410782
            image2[:, 1, :, :] = image2[:, 1, :, :] - 0.433645
            image2[:, 2, :, :] = image2[:, 2, :, :] - 0.452793
        else:
            image1[:, 0, :, :] = image1[:, 0, :, :] #- 0.5
            image1[:, 1, :, :] = image1[:, 1, :, :] #- 0.5
            image1[:, 2, :, :] = image1[:, 2, :, :] #- 0.5

            image2[:, 0, :, :] = image2[:, 0, :, :] #- 0.5
            image2[:, 1, :, :] = image2[:, 1, :, :] #- 0.5
            image2[:, 2, :, :] = image2[:, 2, :, :] #- 0.5            

        features1 = self.netFeatures(image1) # first netry is OG image 
        features2 = self.netFeatures(image2)

        image1_list = [image1] #[ tenOne ]
        image2_list = [image2] #[ tenTwo ]

        # creating list of OG images with different sizes based of feature size
        for intLevel in [ 1, 2, 3, 4, 5]:
            print("intlevel = " , intLevel)
            image1_list.append(F.interpolate(input=image1_list[-1], size=(features1[intLevel].shape[2], features1[intLevel].shape[3]), mode='bilinear', align_corners=False))
            image2_list.append(F.interpolate(input=image2_list[-1], size=(features2[intLevel].shape[2], features2[intLevel].shape[3]), mode='bilinear', align_corners=False))    

        flow = None

        # ipdb.set_trace()
        for intLevel in [-1, -2, -3, -4, -5]: #[ -2, -3, -4, -5 ]:
            print("intLevel = " , intLevel)
            intLevel_ = intLevel #- 1
            flow = self.netMatching[intLevel_](image1_list[intLevel], image2_list[intLevel-1], features1[intLevel], features2[intLevel], flow)
            # print("after Matching flow shape", flow.shape)
            flow = self.netSubpixel[intLevel_](image1_list[intLevel], image2_list[intLevel], features1[intLevel], features2[intLevel], flow)
            # print("after Subpixel flow shape", flow.shape)
            flow = self.netRegularization[intLevel_](image1_list[intLevel], image2_list[intLevel], features1[intLevel], features2[intLevel], flow)
            # print("after regularization flow shape", flow.shape)
        

        # return flow #* 20.0 
        return flow, None 
    
    def forward_v2(self, tenOne, tenTwo):
        
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] #- 0.411618
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] #- 0.434631
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] #- 0.454253

        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] #- 0.410782
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] #- 0.433645
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] #- 0.452793

        tenFeaturesOne = self.netFeatures.forward_v2(tenOne)
        tenFeaturesTwo = self.netFeatures.forward_v2(tenTwo)
        print("tenFeaturesOne len: ", len(tenFeaturesOne))
        print("tenFeaturesOne_0 shape: ", tenFeaturesOne[0].shape)
        print("tenFeaturesOne_1 shape: ", tenFeaturesOne[1].shape)
        print("tenFeaturesOne_2 shape: ", tenFeaturesOne[2].shape)
        print("tenFeaturesOne_3 shape: ", tenFeaturesOne[3].shape)
        print("tenFeaturesOne_4 shape: ", tenFeaturesOne[4].shape)
        print("tenFeaturesOne_5 shape: ", tenFeaturesOne[5].shape)
        tenOne = [ tenOne ]
        tenTwo = [ tenTwo ]

        # creating list of OG images with different sizes based of feature size
        for intLevel in [1, 2, 3, 4, 5 ]:
            tenOne.append(torch.nn.functional.interpolate(input=tenOne[-1], size=(tenFeaturesOne[intLevel].shape[2], tenFeaturesOne[intLevel].shape[3]), mode='bilinear', align_corners=False))
            tenTwo.append(torch.nn.functional.interpolate(input=tenTwo[-1], size=(tenFeaturesTwo[intLevel].shape[2], tenFeaturesTwo[intLevel].shape[3]), mode='bilinear', align_corners=False))
        
        print("tenOne_0 shape: ", tenOne[0].shape)
        print("tenOne_1 shape: ", tenOne[1].shape)
        print("tenOne_2 shape: ", tenOne[2].shape)
        print("tenOne_3 shape: ", tenOne[3].shape)
        print("tenOne_4 shape: ", tenOne[4].shape)
        print("tenOne_5 shape: ", tenOne[5].shape)        

        tenFlow = None

        for intLevel in [ -1, -2, -3, -4, -5 ]:
            print("intLevel = " , intLevel)
            tenFlow = self.netMatching[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            print("after Matching tenFlow shape", tenFlow.shape)
            tenFlow = self.netSubpixel[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            print("after Subpixel tenFlow shape", tenFlow.shape)
            tenFlow = self.netRegularization[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            print("after regularization tenFlow shape", tenFlow.shape)
            if intLevel == -4:
                tenFlow_tmp = tenFlow
        
        ipdb.set_trace()

        return tenFlow * 20.0, tenFlow_tmp
    # end
# end
netNetwork = None

##########################################################

def estimate(image1, image2, ultrasound_flag=False):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    
    intHeight = image1.shape[1]
    intWidth = image1.shape[2]

    print("shape of input image: ", intWidth, " ", intHeight)
    
    image1_batch = image1.cuda().view(1, 3, intHeight, intWidth)
    image2_batch = image2.cuda().view(1, 3, intHeight, intWidth)

    # me: making multiple of 32
    w_32 = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    h_32 = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))
    print("processed shape: " , h_32, " " , w_32)

    image1_resized = F.interpolate(input=image1_batch, size=(h_32, w_32), mode='bilinear', align_corners=False)
    image2_resized = F.interpolate(input=image2_batch, size=(h_32, w_32), mode='bilinear', align_corners=False)
    
    flow, flow_tmp = netNetwork(image1_resized, image2_resized, ultrasound_flag)
    # ipdb.set_trace()
    flow_resized = F.interpolate(input=flow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    flow_resized[:, 0, :, :] *= float(intWidth) / float(w_32)
    flow_resized[:, 1, :, :] *= float(intHeight) / float(h_32)

    return flow_resized[0, :, :, :].cpu() #, flow_tmp[0,:,:,:].cpu()
# end

##########################################################

def plot_flow(flow, img_dim, flow_name, take_flow_size = False, loc= 'images/trials/'):
    # tmp_ = np.zeros((img_dim[1],img_dim[2],3))
    # if take_flow_size:
    # initialize flow size 
    tmp_ = np.zeros((flow.shape[1], flow.shape[2], 3))
    ipdb.set_trace()
    if type(flow) != numpy.ndarray:
        flow_np = flow.permute(1,2,0).numpy() 
    else:
        flow_np = flow #np.transpose(flow, (1,2,0))
        tmp_ = np.zeros((flow.shape[0], flow.shape[1],3))
    # tmp_[..., 1] = 255
    tmp_[..., [0,2]] = cv2.normalize(flow_np,None, 0, 255, cv2.NORM_MINMAX)
    cartesian_bgr = cv2.cvtColor(tmp_.astype(np.uint8), cv2.COLOR_HSV2BGR)    
    cv2.imwrite(os.path.join(loc ,flow_name+'.png'),cartesian_bgr)    

def get_image(i, dataset, PARENT_FOLDER, resize_dim):
    image_path = os.path.join(PARENT_FOLDER, dataset, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    # if not os.path.isabs(image_path):
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)
    # img_tensor = torch.from_numpy(img).permute(2,1,0)
    # img_tensor = img_tensor.type(torch.float32)/255
    return img #img_tensor #torch.from_numpy(img) #img

def find_flow_using_cv2(img1, img2, gaussian_blur_flag = False):
    # ipdb.set_trace()
    # img1_ =  np.transpose(img1, (1,2,0))
    # img2_ = np.transpose(img2, (1,2,0))
    # apply gaussian blue
    if gaussian_blur_flag:
        img1 = cv2.GaussianBlur(img1, (5,5), sigmaX=3) #cv2.BORDER_DEFAULT
        img2 = cv2.GaussianBlur(img2, (5,5), sigmaX=3)
    flow = cv2.calcOpticalFlowFarneback(img1[:,:,0], img2[:,:,0], None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    print("max value of flow = " , np.max(flow) )
    return flow #torch.from_numpy(flow).permute(2,1,0)
    
def get_ultrasound_images(test_dim = (256,256),loc= 'images/trials/'):
    image_ind = [1,2]
    img_list = []
    PARENT_FOLDER = PARENT_FOLDER_TRAIN
    dataset_name = LIST_OF_DATASETS_TRAIN[0]
    for ind_ in image_ind:
        print(ind_)
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(ind_).zfill(6) + '.PNG')
        print(image_path)
        # ipdb.set_trace()
        img = get_image(ind_, dataset_name, PARENT_FOLDER, test_dim) # img already tensor        
        cv2.imwrite(os.path.join(loc, 'US_{}.png'.format(ind_)), img)
        img_list.append(img) 

    return img_list[0], img_list[1]   


if __name__ == '__main__':
    
    # test for RGB image when smaller network used: remove first layer of feature extractor and last layer of pyramid 
    # resize image to 256/ 512 | shape should be independent of netwrk as no bottleneck layer 
    # image1_torch = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # image2_torch = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    #? we start here 
    # image1 = numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].astype(numpy.float32)
    # image2 = numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].astype(numpy.float32)
    # OG_dim = image1.shape
    # # resize images (256,256)
    # test_dim = (256,256)
    # image1_resized = cv2.resize(image1, dsize=test_dim, interpolation=cv2.INTER_NEAREST)
    # image2_resized = cv2.resize(image2, dsize=test_dim, interpolation=cv2.INTER_NEAREST)

    # img1_torch = torch.from_numpy(image1_resized).permute(2,1,0)
    # img2_torch = torch.from_numpy(image2_resized).permute(2,1,0)

    # flow = estimate(img1_torch, img2_torch) 
    # # # # plot flow 
    # plot_flow(flow, OG_dim, 'color_img_extra_layer') 
    # # compare this with cv2.flow and plot 
    # img_flow_cv2 = find_flow_using_cv2(image1, image2)
    # plot_flow(img_flow_cv2, OG_dim, 'color_img_cv2')
    
    ipdb.set_trace()
    #? let's try  on US images 
    test_dim = (256,256)
    img1, img2 = get_ultrasound_images(test_dim)
    ipdb.set_trace()
    img1_torch = torch.from_numpy(img1).permute(2,1,0).type(torch.float32)
    img2_torch = torch.from_numpy(img2).permute(2,1,0).type(torch.float32)

    flow = estimate(img1_torch, img2_torch, True)
    plot_flow(flow, test_dim, 'US_img_extra_layer')

    # compare with cv2.flow
    img_flow_cv2 = find_flow_using_cv2(img1, img2, gaussian_blur_flag=False)
    plot_flow(img_flow_cv2, test_dim, 'US_img_gauss_blur_cv2')
    ipdb.set_trace()

    # start from second feature extractor layer and compare the flow
    # use this to flow in twostream 

    objOutput = open(arguments_strOut, 'wb')

    numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

    # plotting flow in img form 
    plot_flow(flow, OG_dim, 'img_trial')    
    plot_flow(tenOutput_tmp, OG_dim, 'img_tmp_trial', take_flow_size = True)
    plot_flow(output_US, OG_dim, 'img_trial_US')

    ipdb.set_trace()
    plot_flow(output_US_tmp, OG_dim, 'img_tmp_trial_US', take_flow_size= True)
    
    # also find flow using CV2
    US_flow_cv2 = find_flow_using_cv2(img_list[0], img_list[1])
    print("flow shape : " , US_flow_cv2.shape)
    plot_flow(US_flow_cv2, OG_dim ,'img_cv2_trial_US', take_flow_size=True)
    # tmp_ = np.zero_like(tenOne)
    # tenOutput_np = tenOutput.numpy()
    # tmp_[..., 1] = 255
    # tmp_[..., [0,2]] = cv2.normalize(tenOutput_np,None, 0, 255, cv2.NORM_MINMAX)
    # cartesian_bgr = cv2.cvtColor(tmp_, cv2.COLOR_HSV2BGR)    
    # cv2.imwrite('flow_bgr.png',cartesian_bgr)

    objOutput.close()
    ipdb.set_trace()
# end