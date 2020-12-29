import torch
import torch.nn as nn


class VoxRes(nn.Module):
    """
    VoxRes module
    """
    def __init__(self, in_channel):
        super(VoxRes, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channel), nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channel), nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return self.block(x)+x
    
    
class VoxResNet(nn.Module):
    """
    Main model
    """
    def __init__(self, in_channels, num_class):
        super(VoxResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1)
            )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )

        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )
        
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )
        
        self.deconv_c1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, num_class, kernel_size=1))
        
        self.deconv_c2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, num_class, kernel_size=1))
        
        self.deconv_c3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.Conv3d(64, num_class, kernel_size=1))
        
        self.deconv_c4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 8, 8), stride=(1, 8, 8)),
            nn.Conv3d(64, num_class, kernel_size=1))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        
        c1 = self.deconv_c1(out1)
        c2 = self.deconv_c2(out2)
        c3 = self.deconv_c3(out3)
        c4 = self.deconv_c4(out4)
        
        return c1+c2+c3+c4
