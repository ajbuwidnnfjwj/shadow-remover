import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from Model import Generator_F2S, Generator_S2F, Discriminator
from ImageSet import ImageFolder

import matplotlib.pyplot as plt

import itertools
import random

import cv2 as cv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 하이퍼파라미터 설정
BATCH_SIZE = 10
epochs = 300
learning_rate = 0.002
image_channels = 3
guide_channels = 4

# 손실 함수 정의
criterion_adversarial = nn.MSELoss()  # Adversarial loss
criterion_identity = nn.L1Loss()    # identity_loss
criterion_cycle = nn.L1Loss()

# 모델 초기화
generator_f2s = Generator_F2S(guide_channels, image_channels).to(device)
generator_s2f = Generator_S2F(image_channels, image_channels).to(device)
discriminator_s = Discriminator(image_channels).to(device)
discriminator_f = Discriminator(image_channels).to(device)

# 옵티마이저 정의
optimizer_G = optim.Adam(itertools.chain(generator_f2s.parameters(), generator_s2f.parameters()),
                          lr=0.0002, betas=(0.5, 0.999))
optimizer_Df = optim.Adam(discriminator_f.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Ds = optim.Adam(discriminator_s.parameters(), lr=0.0002, betas=(0.5, 0.999))

mask_n = torch.zeros((BATCH_SIZE, 1, 256, 256)).to(device)
label_ones = torch.ones((BATCH_SIZE, 1), requires_grad=False).to(device)
label_zeros = torch.zeros((BATCH_SIZE, 1), requires_grad=False).to(device)

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
    shadow = gray(shadow) * 255
    free = gray(free) * 255
    shadow_np = shadow.cpu().detach().numpy().astype(np.uint8)
    free_np = free.cpu().detach().numpy().astype(np.uint8)
    mask = []
    for i in range(BATCH_SIZE):
        diff =  free_np[i].squeeze(0) - shadow_np[i].squeeze(0)
        _, otsu = cv2.threshold(diff, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask.append(otsu)
    res = torch.FloatTensor(mask).to(device).unsqueeze(1)
    return res / 255

mask_queue = []
max_len = 10

# 훈련 루프
for epoch in range(epochs):
    for i, (shadow, free , mask) in enumerate(dataloader):
        shadow = shadow.to(device)
        free = free.to(device)

        optimizer_G.zero_grad()

        # shadow free identity loss
        free_fake_identity = generator_s2f(free)
        identity_loss_free = criterion_identity(free, free_fake_identity)

        # identity loss shadow
        shadow_fake_identity = generator_f2s(shadow, mask_n)
        identity_loss_shadow = criterion_identity(shadow_fake_identity)

        # shadow cycle consistency loss && shadow free adversarial loss
        free_fake = generator_s2f(shadow)
        mask_queue.append(mask_generator(shadow, free_fake))

        if len(mask_queue) > max_len:
            mask_queue.pop(0)

        output = discriminator_f(free_fake)
        gan_loss_free = criterion_adversarial(label_zeros, output)

        recovered_shadow = generator_f2s(free_fake, mask_queue[-1])
        shadow_cycle_consistency_loss = criterion_cycle(shadow, recovered_shadow)

        # shadow free cycle consistency loss && shadow adversarial loss
        shadow_fake = generator_f2s(free, mask_queue[random.randint(0, len(mask_queue) - 1)])

        output = discriminator_s(shadow_fake)
        gan_loss_shadow = criterion_adversarial(label_zeros, output)

        recovered_free = generator_s2f(shadow_fake)
        free_cycle_consistency_loss = criterion_cycle(free, recovered_free)

        ##################################################################################
        gen_loss = (identity_loss_free + identity_loss_shadow
                      + gan_loss_free + gan_loss_shadow
                      + shadow_cycle_consistency_loss + free_cycle_consistency_loss)
        gen_loss.backward()

        ##################################################################################
        optimizer_Ds.zero_grad()

        pred_real = discriminator_s(shadow)
        label = torch.ones_like(pred_real, requires_grad=False).to(device)
        loss_Ds_real = criterion_adversarial(pred_real, label)

        pred_fake = discriminator_s(shadow_fake.detach())
        label = torch.zeros_like(pred_fake, requires_grad=False).to(device)
        loss_Ds_fake = criterion_adversarial(pred_fake, label)
        loss_Ds = loss_Ds_real + loss_Ds_fake
        loss_Ds.backward()

        ##################################################################################
        optimizer_Df.zero_grad()

        pred_real = discriminator_f(free)
        label = torch.ones_like(pred_real, requires_grad=False).to(device)
        loss_Df_real = criterion_adversarial(pred_real, label)

        pred_fake = discriminator_f(free_fake.detach())
        label = torch.zeros_like(pred_fake, requires_grad=False).to(device)
        loss_Df_fake = criterion_adversarial(pred_fake, label)
        loss_Df = loss_Df_real + loss_Df_fake
        loss_Df.backward()

        if (i+1) % 10 == 0:
            print(f'{epoch} / {epochs}, batch {i+1}/{len(dataloader)}', end = ' ')
            print(f'{gen_loss.item():.4f}')
            transform = transforms.ToPILImage()
            plt.imshow(transform(free_fake[0]))
            plt.show()

torch.save(generator_f2s.state_dict(), 'models/generator_f2s.pth')
torch.save(generator_s2f.state_dict(), 'models/generator_s2f.pth')
torch.save(discriminator_s.state_dict(), 'models/discriminator_s.pth')
torch.save(discriminator_f.state_dict(), 'models/discriminator_f.pth')