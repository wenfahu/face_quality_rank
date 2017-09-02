from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

class SiameseData(Dataset):
    def __init__(self, image_paths0, image_paths1, labels):

