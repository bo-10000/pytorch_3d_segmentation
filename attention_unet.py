import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)
    

class Upsample(nn.Module):
    """
    upsample, concat and conv
    """
    def __init__(self, in_channel, inter_channel, out_channel):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channel, inter_channel),
            nn.Upsample(scale_factor=2)
        )
        self.conv = ConvBlock(2*inter_channel, out_channel)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out


class AttentionGate(nn.Module):
    """
    filter the features propagated through the skip connections
    """
    def __init__(self, in_channel, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv3d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv3d(in_channel, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv3d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out, size=x.size()[2:], mode='trilinear')
        out = x * out
        return out
    
    
class AttentionUNet(nn.Module):
    """
    Main model
    """
    def __init__(self, in_channel, num_class, filters=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        f1, f2, f3, f4 = filters
        
        self.down1 = ConvBlock(in_channel, f1)
        
        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f1, f2)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f2, f3)
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f3, f4)
        )

        self.ag1 = AttentionGate(f3, f4, f3)
        self.ag2 = AttentionGate(f2, f2, f2)
        self.ag3 = AttentionGate(f1, f1, f1)
        
        self.up1 = Upsample(f4, f3, f2)
        self.up2 = Upsample(f2, f2, f1)
        self.up3 = Upsample(f1, f1, num_class)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        ag1 = self.ag1(down3, down4)
        up1 = self.up1(down4, ag1)
        ag2 = self.ag2(down2, up1)
        up2 = self.up2(up1, ag2)
        ag3 = self.ag3(down1, up2)
        up3 = self.up3(up2, ag3)
        
        return up3
