import torch
import argparse
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataloader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd

from densenet import densenet121

def main(args):

    transform_train = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.ToTensor()
        ])

    anno = pd.read_csv(args.anno_txt, header=None)


    train_set = SiameseData(args.data_dir, image_paths0,
            image_paths1, targets, transform=transform_train)
    train_loader = Dataloader(train_set, batch_size=
            args.batch_size,
            shuffle=True)
    model = densenet121(drop_rate=args.drop_rate)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()
    if args.ckpt:
        if os.path.isfile(args.ckpt):
            print('loading ckpt {}'.format(args.ckpt))
            checkpoint = torch.load(args.ckpt)
            state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k in model.state_dict().keys():
                    state_dict[k] = v
            model.load_state_dict(state_dict)
            print('checkpoint loaded')

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
            weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        # adjust learning rate


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to the training set')
    parser.add_argument('--image_size', type=int,  help='training image size')
    parser.add_argument('--training_dir', help='path to store the ckpts')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()
    main(args)

