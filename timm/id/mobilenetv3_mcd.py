import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import numpy as np
from collections import deque
from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride


        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):  
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MNV3_large2(nn.Module):
    def __init__(self, numclasses=2):
        super(MNV3_large2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=1),
            #nn.Dropout2d(0.2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=2),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=32,
                  nolinear=hswish(), semodule=SeModule(32), stride=1),
            #nn.Dropout2d(0.3),
            Block(kernel_size=3, in_size=32, expand_size=128, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),
            #nn.Dropout2d(0.5),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=2),
            Block(kernel_size=3, in_size=48, expand_size=192, out_size=48,
                  nolinear=hswish(), semodule=SeModule(48), stride=1),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.5),
        )
        self.GRL=GRL()
        self.classifier1 = nn.Conv2d(in_channels=48, out_channels=numclasses, kernel_size=1,
                      stride=1, padding=0, bias=True)
        self.classifier2 = nn.Conv2d(in_channels=48, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=True)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        feature = self.bneck(x)
        feature = feature.view(feature.size(0), -1,1,1)
        return feature

class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.
    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self,num_classes=3,in_channel=48):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.classifier = nn.Conv2d(in_channels=in_channel , out_channels=num_classes, kernel_size=1,
              stride=1, padding=0, bias=True)
        #self.reverse = reverse
    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, lambd=1, reverse=0):
        x=x.reshape(-1,self.in_channel,1,1)
        if reverse==1:
            #print("grl:lambd={}".format(lambd))
            x=GRL.apply(x,lambd)
        output=self.classifier(x)
        output=output.view(output.size(0), -1)
        return output
class SameClassifierHead(nn.Module):
    r"""Classifier Head for MCD.
    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self,num_classes=2,in_channel=48):
        super(SameClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=64, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=2),
            Block(kernel_size=3, in_size=24, expand_size=96, out_size=24,
                  nolinear=hswish(), semodule=SeModule(24), stride=1),
            #nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.5),
        )
        self.classifier = nn.Conv2d(in_channels=24 , out_channels=4, kernel_size=1,
              stride=1, padding=0, bias=True)
        self.classifier2 = nn.Conv2d(in_channels=36 , out_channels=num_classes, kernel_size=1,
              stride=1, padding=0, bias=True)
        #self.reverse = reverse
    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, lambd=1, reverse=0):
        x=x.reshape(-1,self.in_channel,2,1)
        x=x.permute(0,2,1,3)
        x=x.reshape(-1,2,6,8)
        if reverse==1:
            #print("grl:lambd={}".format(lambd))
            x=GRL.apply(x,lambd)
        feat=self.layer1(x)
        #print(feat.size())
        feat=self.bneck(feat)
        #print(feat.size())
        output=self.classifier(feat)
        #print(output.size())
        output=output.contiguous().view(output.size(0),-1,1,1)
        output=self.classifier2(output)
        output=output.view(output.size(0),-1)
        #print(output.size())
        return output

def test():
    G = MNV3_large2()
    C = ImageClassifierHead()
    x = torch.randn(1024,1,188,188)
    y = G(x)
    y = C(y)
    #print(y.size())
#if __name__ == '__main__':

    # pthpath = r'test_acc_99.6159936658749.pth'
    # net = MNV3_small_small(2)
    # net.load_state_dict(torch.load(pthpath).state_dict())
    # net.to('cpu')
    # net.eval()
    #
    # for i in net.modules():#childerens
    #     if not isinstance(i, nn.Sequential) and not isinstance(i, Block):
    #         # print(i)
    #         pass
    # a = torch.ones((1, 1, 128, 128))
    # c = net(a)
    #print(net.named_parameters())
    # ppp=0
    # total = 0
#test()
