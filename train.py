import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from Model import Generator, Discriminator
from ImageSet import ImageFolder

import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라미터 설정
batch_size = 30
epochs = 100
learning_rate = 0.002
image_size = (3, 256, 256)

# 손실 함수 정의
criterion_GAN = nn.MSELoss()  # Adversarial loss
creterion_identity = nn.L1Loss()    # identity_loss

# 모델 초기화
generator_f = Generator(image_size).to(device)
generator_s = Generator((4,256,256)).to(device)
discriminator = Discriminator(image_size).to(device)

# 옵티마이저 정의
optimizer_G = optim.Adam(list(generator_f.parameters())+list(generator_s.parameters()),
                          lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기 조정
    transforms.ToTensor()
])
dataset = ImageFolder(root='images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 훈련 루프
for epoch in range(epochs):
    for i, (shadow, mask) in enumerate(dataloader):
        shadow = shadow.to(device)
        mask = mask.to(device)

        # ---------------------
        # Generator 훈련
        # ---------------------
        optimizer_G.zero_grad()

        # 그림자 제거 이미지 생성
        shadow_free = generator_f(shadow)
        Gs_input = torch.cat((shadow_free, mask), 1)
        shadow_fake = generator_s(Gs_input)

        identity_loss = creterion_identity(shadow, shadow_fake)
        identity_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Discriminator 훈련
        # ---------------------
        optimizer_D.zero_grad()
        predicted = discriminator(shadow_free.detach())

        label = torch.zeros_like(predicted)

        adversarial_loss = criterion_GAN(predicted, label)
        adversarial_loss.backward()
        optimizer_D.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Loss G: {identity_loss.item():.4f}, Loss D: {adversarial_loss.item():.4f}")

torch.save(generator_f.state_dict(), 'generator.pth')