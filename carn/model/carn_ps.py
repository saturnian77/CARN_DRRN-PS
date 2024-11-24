import torch
import torch.nn as nn
import model.ops as ops
import numpy as np

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, banklength, matrix):
        super(Block, self).__init__()

        self.b1_x2 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b1_x3 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b1_x4 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b2_x2 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b2_x3 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b2_x4 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b3_x2 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b3_x3 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.b3_x4 = ops.ResidualBlock_u(in_channels, out_channels, banklength, matrix)
        self.c1 = ops.BasicBlock(out_channels*2, out_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(out_channels*3, out_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(out_channels*4, out_channels, 1, 1, 0)

    def forward(self, x, scale):
        c0 = o0 = x

        if scale==2:
            b1 = self.b1_x2(o0)
        elif scale==3:
            b1 = self.b1_x3(o0)
        elif scale==4:
            b1 = self.b1_x4(o0)
        

        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        if scale==2:
            b2 = self.b2_x2(o1)
        elif scale==3:
            b2 = self.b2_x3(o1)
        elif scale==4:
            b2 = self.b2_x4(o1)
        
        
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        if scale==2:
            b3 = self.b3_x2(o2)
        elif scale==3:
            b3 = self.b3_x3(o2)
        elif scale==4:
            b3 = self.b3_x4(o2)

        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)
        gamma = 8/9 # 6/9 or 8/9

        filt_num = np.round(162*gamma).astype(int) # 1x1x64x64 kernel partitioning, 3RB * 3B * 2layer * 3x3 kernel, compression rate 6/9 = 18*9*(6/9)  = 108, 8/9=144
        self.filterbank = nn.Parameter(torch.randn(filt_num,64*64), requires_grad=True)
        nn.init.kaiming_normal_(self.filterbank)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64, filt_num, self.filterbank)
        self.b2 = Block(64, 64, filt_num, self.filterbank)
        self.b3 = Block(64, 64, filt_num, self.filterbank)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = ops.UpsampleBlock(64, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)


                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0,scale)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1,scale)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2,scale)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
