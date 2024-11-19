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
BATCH_SIZE = 10
epochs = 300
learning_rate = 0.002
image_size = (3, 256, 256)

# 손실 함수 정의
criterion_adversarial = nn.MSELoss()  # Adversarial loss
criterion_consistency = nn.L1Loss()    # identity_loss

# 모델 초기화
generator_f = Generator(image_size).to(device)
generator_s = Generator(image_size).to(device)
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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
        guided = torch.cat([shadow_free, mask], dim=1)
        shadow_fake = generator_s(shadow_free, mask)

        cycle_consist_loss = criterion_consistency(shadow_fake, shadow)
        cycle_consist_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Discriminator 훈련
        # ---------------------
        optimizer_D.zero_grad()
        predicted = discriminator(outed_mask.detach())

        label = torch.zeros_like(predicted)

        discriminate_loss = criterion_consistency(predicted, label)

        predicted = discriminator(mask.detach())
        label = torch.ones_like(predicted)

        discriminate_loss += criterion_consistency(predicted, label)
        discriminate_loss.backward()
        optimizer_D.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(dataloader)}]"
                  f"generator loss {mask_loss.item():.4f}, discriminator loss: {discriminate_loss.item():.4f}")
            transform = transforms.ToPILImage()
            plot_free = transform(shadow_free[0])
            plot_mask = transform(outed_mask[0])

            # 서브플롯 구성 (2행 3열)
            fig, axes = plt.subplots(1, 2)

            axes[0].imshow(plot_free)
            axes[0].axis('off')
            axes[0].set_title('shadow free')
            axes[1].imshow(plot_mask)
            axes[1].axis('off')
            axes[1].set_title('mask')

            plt.tight_layout()
            plt.show()

torch.save(generator_f.state_dict(), 'generator.pth')