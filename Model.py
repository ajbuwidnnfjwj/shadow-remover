import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

class Generator(nn.Module):
    '''UNet based Generator'''
    def __init__(self, size):
        '''
        :param size: tuple for size of input image (channels, height, width)
        '''
        super(Generator, self).__init__()
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

# 판별기 정의
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters():  # 합성곱 레이어는 초기화하지 않음
            param.requires_grad = False
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )


    def forward(self, img):
        '''
        :param img: A batch of input images
        :return:
        '''
        return self.model(img)