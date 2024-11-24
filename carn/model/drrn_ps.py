import torch
import torch.nn as nn
import model.ops as ops 
import torch.nn.functional as F

def rgb_to_ycbcr(input):
  # input is mini-batch N x 3 x H x W of an RGB image
  output = input.clone()
  input = input*255.
  output[:, 0, :, :] = input[:, 0, :, :] * 65.481/255 + input[:, 1, :, :] * 128.553/255 + input[:, 2, :, :] * 24.966/255 + 16
  output[:, 1, :, :] = input[:, 0, :, :] * -37.797/255 + input[:, 1, :, :] * -74.203/255 + input[:, 2, :, :] * 112./255 + 128
  output[:, 2, :, :] = input[:, 0, :, :] * 112./255 + input[:, 1, :, :] * -93.786/255 + input[:, 2, :, :] * -18.214/255 + 128
  output = ((output-16)/219.0 - 0.5) * 2.0

  return output

def ycbcr_to_rgb(input):
  # input is mini-batch N x 3 x H x W of an RGB image
  output = input.clone()
  input = ((input + 1.0)*219. / 2.0)+16
  output[:, 0, :, :] = input[:, 0, :, :] * 0.00456621*255 + 0                                 + input[:, 2, :, :] * 0.00625893*255 -222.921
  output[:, 1, :, :] = input[:, 0, :, :] * 0.00456621*255 + input[:, 1, :, :] *  -0.00153632*255 + input[:, 2, :, :] * -0.00318811*255 + 135.576
  output[:, 2, :, :] = input[:, 0, :, :] * 0.00456621*255 + input[:, 1, :, :] * 0.00791071*255 + input[:, 2, :, :] * 0 -276.836
  output = output/255.0
  return output

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, banklength, matrix):
        super(Block, self).__init__()

        self.b1_x2 = ops.ResidualBlock_drrn(in_channels, out_channels, banklength, matrix)
        self.b1_x3 = ops.ResidualBlock_drrn(in_channels, out_channels, banklength, matrix)
        self.b1_x4 = ops.ResidualBlock_drrn(in_channels, out_channels, banklength, matrix)


    def forward(self, x, scale):

        if scale==2:
            b1 = self.b1_x2(x)
        elif scale==3:
            b1 = self.b1_x3(x)
        elif scale==4:
            b1 = self.b1_x4(x)

        
        return b1
        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)
          
        filt_num = 18 # 1x1x128x128 kernel partitioning, 9block, 2 layer, 3x3 kernel,recursive(1/9 innate reduction rate)
        self.filterbank = nn.Parameter(torch.randn(filt_num,128*128), requires_grad=True) # 1x1 kernel partitioning
        nn.init.kaiming_normal_(self.filterbank)

        
        self.entry = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.b1 = Block(128, 128, filt_num, self.filterbank)
        self.b2 = Block(128, 128, filt_num, self.filterbank)
        self.b3 = Block(128, 128, filt_num, self.filterbank)
        self.b4 = Block(128, 128, filt_num, self.filterbank)
        self.b5 = Block(128, 128, filt_num, self.filterbank)
        self.b6 = Block(128, 128, filt_num, self.filterbank)
        self.b7 = Block(128, 128, filt_num, self.filterbank)
        self.b8 = Block(128, 128, filt_num, self.filterbank)
        self.b9 = Block(128, 128, filt_num, self.filterbank)
        
        self.outact = nn.ReLU()
        self.exit = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)


                
    def forward(self, x, scale):
        x_up = F.interpolate(x, scale_factor=scale,mode='bicubic')
        x_up = x_up.clamp(min=0.,max=1.)
        ycbcr = rgb_to_ycbcr(x_up.float())
        inp = ycbcr[:,0:1,:,:]

        ##


        res = self.entry(inp)

        res_= self.b1(res,scale)+res
        res_= self.b2(res_,scale)+res
        res_= self.b3(res_,scale)+res
        res_= self.b4(res_,scale)+res
        res_= self.b5(res_,scale)+res
        res_= self.b6(res_,scale)+res
        res_= self.b7(res_,scale)+res
        res_= self.b8(res_,scale)+res
        res_= self.b9(res_,scale)+res
        
        ##

        out = torch.add(self.exit(self.outact(res_)),inp)
        out = ycbcr_to_rgb(torch.cat((out,ycbcr[:,1:3,:,:]),1))  

        return out
