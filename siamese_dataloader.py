from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class SiameseData(Dataset):
    def __init__(self, transform=None, image_paths0, image_paths1, labels, loader=default_laoder):
        self.image_pairs =zip(image_paths0, image_paths1)
        self.labels = labels

    def __getitem__(self, index):
        path0, path1 = self.image_pairs[index]
        img0 = self.loader(path0)
        img1 = self.loader(path1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        target = self.labels[index]
        return img0, img1, torch.LongTensor(target)

    def __len__(self):
        return len(self.labels)

