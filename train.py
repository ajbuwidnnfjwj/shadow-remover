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
batch_size = 64
epochs = 300
learning_rate = 0.002
image_size = (3, 256, 256)

generator = Generator(image_size).to(device)
discriminator = Discriminator(image_size).to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기 조정
    transforms.ToTensor()
])
dataset = ImageFolder(root='images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 훈련 루프
for epoch in range(epochs):
    for i, (shadowed_images, real_images) in enumerate(dataloader):
        shadowed_images = shadowed_images.to(device)
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 진짜 및 가짜 레이블 생성
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # 판별기 훈련
        optimizer_D.zero_grad()

        # 진짜 이미지 손실 계산
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        # 가짜 이미지 생성 및 손실 계산
        fake_images = generator(shadowed_images)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)


        # 판별기 손실과 업데이트
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 생성기 훈련
        optimizer_G.zero_grad()

        # 생성된 이미지 손실 계산
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # 진행 상황 출력
        if i % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            transform = transforms.Compose([
                transforms.ToPILImage()
            ])
            plt.imshow(transform(fake_images[0]))
            plt.axis('off')
            plt.show()

torch.save(generator.state_dict(), 'generator.pth')