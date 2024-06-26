import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
class GSNorm3d(torch.nn.Module):
    def __init__(self, out_ch, num_group=1):
        super().__init__()
        self.out_ch = out_ch
        self.num_group=num_group
        #self.activation = nn.ReLU()
    def forward(self, x):
        interval = self.out_ch//self.num_group
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            #dominator = torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)
            #dominator = dominator + (dominator<0.001)*1
            tensors.append(x[:,start_index:start_index+interval,...]/(torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)+0.0001))
            start_index = start_index+interval
        
        return torch.cat(tuple(tensors),dim=1)
def Normalization(norm_type, out_channels,num_group=1):
    if norm_type==1:
        return nn.InstanceNorm2d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm2d(out_channels,momentum=0.1)
    elif norm_type==3:
        return GSNorm2d(out_channels,num_group=num_group)
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),     
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation,
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, num_group=1,activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2),stride=(2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2),stride=(2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class VAE(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_classes, bilinear,norm_type=2, n_fmaps=[8,16,32,64,128,256],dim=1024,soft=False):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type,soft=False)
        self.fc_mean = torch.nn.Linear(2304,dim)
        self.fc_std = torch.nn.Linear(2304,dim)
        self.fc2 = torch.nn.Linear(dim,2304)
        self.up1 = Up(n_fmaps[5],n_fmaps[4],norm_type=norm_type,soft=False)
        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv2d(8, n_classes, 3, padding=1)
        self.final = nn.Softmax(dim=1)
    def forward(self, x,if_random=False,scale=1,mid_input=False,dropout=0.0):
        #'pred_only','pred_recon',if_random=False
        #x = data_dict[in_key]
        
        if not mid_input:
            #input_res = data_dict.get(self.in_key2)
            #input_x = self.SConv3d(input_x)
            x = self.in_block(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = x.view(x.size(0),x.size(1)*x.size(2)*x.size(3))
            x_mean = self.fc_mean(x)
            x_std = nn.ReLU()(self.fc_std(x))
            #data_dict['mean'] = x_mean
            #data_dict['std'] = x_std
            z = torch.randn(x_mean.size(0),x_mean.size(1)).type(torch.cuda.FloatTensor)
            if if_random:
                x = self.fc2(x_mean+z*x_std*scale)
            else:
                x = self.fc2(x_mean)
        else:
            x = self.fc2(x)
        x = x.view(x.size(0),256,3,3)
        
        x = self.up1(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x=F.interpolate(x, size=(x.size(2)+1,x.size(3)+1), mode='nearest')
        x = self.up2(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        #x = self.final(x)
        #data_dict[out_key] = x
        if not mid_input:
            #return x,x_mean,x_std
            return x
        else:
            return x