""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.double_block = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.Conv2d(128,128,kernel_size=1))#扩充通道然后分成两份
        self.heatmap_predictor=nn.Sequential(nn.Conv2d(64, 256, kernel_size=3,padding=1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                                    nn.Conv2d(256, 128, kernel_size=3,padding=1,bias=False),
                                                    nn.BatchNorm2d(128),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(128,3, kernel_size=1,padding=0))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        double_x=self.double_block(x)
        num_channels = double_x.shape[1]
        half_channels = num_channels // 2
        x_1=double_x[:,:half_channels,:,:]#分成两部分
        x_2=double_x[:,half_channels:,:,:]
        logits = self.outc(x_1)
        heatmap=self.heatmap_predictor(x_2)
        return logits,heatmap,x1,x2,x3,x4,x5

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)