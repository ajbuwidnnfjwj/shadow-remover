import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

class UNet(nn.Module):
    def __init__(self, size):
        super(UNet, self).__init__()
        # 인코더
        self.enc1 = self.conv_block(size[0], 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 디코더
        self.dec1 = self.up_conv(512, 256)
        self.dec2 = self.up_conv(256, 128)
        self.dec3 = self.up_conv(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # 인코더 단계
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # 디코더 단계
        d1 = self.dec1(e4)
        d2 = self.dec2(d1 + e3)
        d3 = self.dec3(d2 + e2)
        out = self.final_conv(d3 + e1)

        return out

class Generator(nn.Module):
    def __init__(self, size):
        super(Generator, self).__init__()
        self.model = UNet(size)

    def forward(self, z):
        return self.model(z)

# 판별기 정의
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        input_size = image_size[0] * image_size[1] * image_size[2]
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)