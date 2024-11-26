from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.path_noise = [root+'shadow/'+fname for fname in os.listdir(root+'shadow')]
        self.path_removed = [root + 'removed/' + fname for fname in os.listdir(root + 'removed')]
        self.path_mask = [root + 'mask/' + fname for fname in os.listdir(root + 'mask')]

    def __len__(self):
        return len(self.path_noise) \
            if len(self.path_noise) == len(self.path_removed) == len(self.path_mask) else False

    def __getitem__(self, idx):
        image_path_noise = self.path_noise[idx]
        image_path_removed = self.path_removed[idx]
        image_path_mask = self.path_mask[idx]
        noise = Image.open(image_path_noise)
        removed = Image.open(image_path_removed)
        mask = Image.open(image_path_mask)

        if self.transform:
            noise = self.transform(noise)
            removed = self.transform(removed)
            mask = self.transform(mask)

        return noise, removed, mask