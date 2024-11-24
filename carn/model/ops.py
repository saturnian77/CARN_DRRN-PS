import math
import torch
import torch.nn as nn
from   torch import einsum
import torch.nn.init as init
import torch.nn.functional as F
from   einops import rearrange, repeat
from   einops.layers.torch import Rearrange, Reduce

def init_weights(modules):
    pass
   

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class conv_KF3x3(nn.Module):
    def __init__(self, ch_in, ch_out, num, matrix):
        super().__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.matrix = matrix
        self.num = num
        self.init = True
        self.element = nn.Parameter(torch.randn(9, num), requires_grad=True) #1x1x64x64
        self.bias = nn.Parameter(torch.randn(ch_out), requires_grad=True)
            
        if self.init == True:
            self.initialize()
                
    def initialize(self):
        nn.init.constant_(self.element,1/self.num)
        nn.init.constant_(self.bias,0)
        self.init = False
        
    def forward(self, x):
        self.kernel = rearrange(torch.matmul(self.element,self.matrix), '(k1 k2) (cin cout) -> cout cin k1 k2', cin = self.ch_in, cout = self.ch_out, k1=3, k2=3) # 1x1x64x64
        out = F.conv2d(input = x, weight = self.kernel, bias = self.bias, stride = 1, padding = 1)
        return out

class conv_KF3x3_nb(nn.Module):
    def __init__(self, ch_in, ch_out, num, matrix):
        super().__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.matrix = matrix
        self.num = num
        self.init = True
        self.element = nn.Parameter(torch.randn(9, num), requires_grad=True) #1x1x64x64
        #self.bias = nn.Parameter(torch.randn(ch_out), requires_grad=True)
            
        if self.init == True:
            self.initialize()
                
    def initialize(self):
        nn.init.constant_(self.element,1/self.num)
        #nn.init.constant_(self.bias,0)
        self.init = False
        
    def forward(self, x):
        self.kernel = rearrange(torch.matmul(self.element,self.matrix), '(k1 k2) (cin cout) -> cout cin k1 k2', cin = self.ch_in, cout = self.ch_out, k1=3, k2=3) # 1x1x64x64
        out = F.conv2d(input = x, weight = self.kernel, bias = None, stride = 1, padding = 1)
        return out

class ResidualBlock_drrn(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, banklength, matrix):
        super(ResidualBlock_drrn, self).__init__()

        self.actin = nn.ReLU()
        self.conv1 = conv_KF3x3_nb(in_channels, out_channels, banklength, matrix)
        self.act    = nn.ReLU()
        self.conv2 = conv_KF3x3_nb(out_channels, out_channels, banklength, matrix)
        

        
    def forward(self, x):
        y = self.actin(x)
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        return y        

class ResidualBlock_u(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, banklength, matrix):
        super(ResidualBlock_u, self).__init__()

        self.conv1 = conv_KF3x3(in_channels, out_channels, banklength, matrix)
        self.act    = nn.ReLU()
        self.conv2 = conv_KF3x3(out_channels, out_channels, banklength, matrix)
        self.actout = nn.ReLU()

        
    def forward(self, x):
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y = self.actout(y + x)
        return y


class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
