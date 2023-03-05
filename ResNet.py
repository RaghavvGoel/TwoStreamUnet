'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel_in=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        #* mask concatenated with image and then encoded. Change in_channels = 4 from 3
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=3, stride=1, padding=1, bias=False) #spatial res maintained
        self.bn1 = nn.BatchNorm2d(64)
        #! mention sizes here | change so that final size is Cx28x28
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # if H,W are odd, then H,H+1 -> H/2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def init_weight(self, pretrained_wgts):
        # don't assign weights to conv1 
        pass

    def forward(self, x):
        # print("x.shape = ", x.shape)
        # ipdb.set_trace()
        features = []
        out = F.relu(self.bn1(self.conv1(x))) # 64 x 220 x 220
        # print("out1: ", out.shape) #* can choose either out1 or out2 for skip connection
        # features.append(out)

        out = self.layer1(out) # 64 x 220 x 220
        # print("out2: ", out.shape) 
        features.append(out)
        
        out = self.layer2(out) # 128 x 110 x 110
        # print("out3: ", out.shape)
        features.append(out)
        
        out = self.layer3(out) # 256 x 55 x 55
        # print("out4: ", out.shape)
        features.append(out)
        
        out = self.layer4(out) # 512 x 28 x 28
        # print("out5: ", out.shape)
        features.append(out)

        out = F.avg_pool2d(out, 4) # 512 x 7 x 7
        # features.append(out)

        # print("out6: ", out.shape)
        # remove linear layers 
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out, features


def ResNet18(channel_in = 3):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel_in=channel_in)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(channel_in=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], channel_in=channel_in)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()