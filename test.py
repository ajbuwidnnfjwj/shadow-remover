import torch
from PIL import Image

import torchvision
from torchvision import transforms

import cv2 as cv
import numpy as np

import Model

img = Image.open('target.jpg')
origin_size = img.size
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

model = Model.Generator((3,256,256))
model.load_state_dict(torch.load("model/generator.pth"))

output = model(transform(img)).detach()
retrans = transforms.Compose([
    transforms.Resize(img.size),
    transforms.ToPILImage()
])
output = retrans(output)
output.show()