from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
import collections

class ReScale(object):
    """Rescale the input PIL.Image to the given size.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        return img.resize(self.size, self.interpolation)


def collate_cat(batch):
    '''concatenate sub-batches along first dimension'''
    if torch.is_tensor(batch[0]):
        return torch.cat(batch,0)
    elif isinstance(batch[0], collections.Iterable):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [collate_cat(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, or lists; found {}"
                     .format(type(batch[0]))))

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
    return pil_loader(path)

class SiameseData(Dataset):
    def __init__(self, root, image_paths0, image_paths1, labels, transform=None, loader=default_loader):
        self.image_pairs =zip(image_paths0, image_paths1)
        self.labels = labels
        self.loader = loader
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        path0, path1 = self.image_pairs[index]
        img0 = self.loader(os.path.join(self.root, path0))
        img1 = self.loader(os.path.join(self.root, path1))
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        target = self.labels[index]
        return img0.unsqueeze(0), img1.unsqueeze(0), torch.LongTensor([target])

    def __len__(self):
        return len(self.labels)
