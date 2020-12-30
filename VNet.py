import torch
import torch.nn as nn


class RepeatConv(nn.Module):
    """
    Repeat Conv + PReLU n times
    """
    def __init__(self, n_channels, n_conv):
        super(RepeatConv, self).__init__()
        
        conv_list = []
        for _ in range(n_conv):
            conv_list.append(nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2))
            conv_list.append(nn.PReLU())
        
        self.conv = nn.Sequential(
            *conv_list
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super(Down, self).__init__()
        
        self.downconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels, n_conv)
        
    def forward(self, x):
        out = self.downconv(x)
        return out + self.conv(out)

    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super(Up, self).__init__()
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, int(out_channels / 2), kernel_size=2, stride=2),
            nn.PReLU()
        )
        self.conv = RepeatConv(out_channels, n_conv)
        
    def forward(self, x, down):
        x = self.upconv(x)
        cat = torch.cat((x, down), dim=1)
        return cat + self.conv(cat)
    

class VNet(nn.Module):
    """
    Main model
    """
    def __init__(self, in_channels, num_class):
        super(VNet, self).__init__()
        
        self.down1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, padding=2),
            nn.PReLU()
        )
        
        self.down2 = Down(16, 32, 2)
        self.down3 = Down(32, 64, 3)
        self.down4 = Down(64, 128, 3)
        self.down5 = Down(128, 256, 3)
        
        self.up1 = Up(256, 256, 3)
        self.up2 = Up(256, 128, 3)
        self.up3 = Up(128, 64, 2)
        self.up4 = Up(64, 32, 1)
        
        self.up5 = nn.Sequential(
            nn.Conv3d(32, num_class, kernel_size=1),
            nn.PReLU()
        )
         
    def forward(self, x):
        down1 = self.down1(x) + torch.cat(16*[x], dim=1)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.down5(down4)
        up1 = self.up1(center, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)
        return self.up5(up4)
