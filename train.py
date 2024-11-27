import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from Model import Generator, Discriminator
from ImageSet import ImageFolder

import matplotlib.pyplot as plt

import itertools

import cv2 as cv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라미터 설정
BATCH_SIZE = 10
epochs = 300
learning_rate = 0.002
image_size = (3, 256, 256)
guide_size = (4, 256, 256)

# 손실 함수 정의
criterion_adversarial = nn.MSELoss()  # Adversarial loss
criterion_identity = nn.L1Loss()    # identity_loss
criterion_cycle = nn.L1Loss()

# 모델 초기화
generator_f2s = Generator(guide_size).to(device)
generator_s2f = Generator(image_size).to(device)
discriminator_s = Discriminator(image_size).to(device)
discriminator_f = Discriminator(image_size).to(device)

# 옵티마이저 정의
optimizer_G = optim.Adam(itertools.chain(generator_f2s.parameters(), generator_s2f.parameters()),
                          lr=0.0002, betas=(0.5, 0.999))
optimizer_DA = optim.Adam(discriminator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_DB = optim.Adam(discriminator_s.parameters(), lr=0.0002, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기 조정
    transforms.ToTensor()
])
dataset = ImageFolder(root='images/', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

import cv2
gray = transforms.Grayscale()
to_pil = transforms.ToPILImage()
def mask_generator(shadow, free):
    shadow = gray(shadow).squeeze(0) * 255
    free = gray(free).squeeze(0) * 255
    shadow_np = shadow.cpu().numpy().astype(np.uint8)
    free_np = free.cpu().numpy().astype(np.uint8)

    diff =  free_np-shadow_np
    _, otsu = cv2.threshold(diff, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return torch.from_numpy(otsu).float() / 255

# 훈련 루프
for epoch in range(epochs):
    for i, (shadow, free , mask) in enumerate(dataloader):
        shadow = shadow.to(device)
        free = free.to(device)
        mask = mask.to(device)

        # ---------------------
        # Generator 훈련
        # ---------------------
        optimizer_G.zero_grad()

        #identity loss
        free_fake = generator_s2f(free) # ~If
        identity_loss_free = criterion_identity(free, free_fake)

        #GAN loss
        output = discriminator_f(free_fake)
        label = torch.ones_like(free_fake, requires_grad=False).to(device)
        gan_loss_free = criterion_adversarial(label, output)

        guided = torch.cat((free, mask), dim=1)
        shadow_fake = generator_f2s(guided) # ~Is
        output = discriminator_s(shadow_fake)
        label = torch.zeros_like(output, requires_grad=False).to(device)
        gan_loss_shadow = criterion_adversarial(label, output)

        #shadow cycle_consistency
        guided = torch.cat((free_fake, mask), dim=1)
        shadow_fake_cycle = generator_f2s(guided)
        consistency_loss_shadow = criterion_cycle(shadow_fake_cycle, shadow)
