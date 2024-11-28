import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.InstanceNorm2d(channel),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    '''UNet based Generator'''
    def __init__(self, size):
        '''
        :param size: tuple for size of input image (channels, height, width)
        '''
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(size[0], 64, kernel_size = 3, stride = 2),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(9):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_features, 3, 7)]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return x+self.model(x)


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