import torch
import argparse
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd
# from visdom import Visdom

from densenet import densenet121
from rank_loss import RankLoss
from siamese_dataloader import SiameseData, collate_cat, ReScale



def main(args):

    transform_train = transforms.Compose([
        ReScale((args.image_size, args.image_size)),
        transforms.ToTensor()
        ])

    anno = pd.read_csv(args.anno_txt, sep=';', header=None)
    image_paths0 = list(anno.ix[:, 1])
    image_paths1 = list(anno.ix[:, 2])
    targets = np.array(anno.ix[:,3])
    targets = np.sign(targets - 3)
    global plotter 
    plotter = VisdomLinePlotter(env_name=args.name)


    train_set = SiameseData(args.data_dir, image_paths0,
            image_paths1, targets, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=
            args.batch_size, #num_workers=1, # pin_memory=True,
            collate_fn = collate_cat,
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
    criterion = RankLoss()
    for epoch in range(args.epoch):
        # adjust learning rate
        train(train_loader, model, criterion, optimizer, epoch)
        save_ckpt(model, args.training_dir, epoch)

def save_ckpt(model, train_dir, epoch):
    filename = os.path.join(train_dir, 'checkpoint.t7')
    state = {
            'epoch': epoch,
            'state': model.state_dict()
            }
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    for idx, (x0, x1, t) in enumerate(train_loader):
        target = t.cuda(async=True)
        x0 = x0.cuda()
        x1 = x1.cuda()
        x0_var = Variable(x0)
        x1_var = Variable(x1)
        target_var = Variable(target)
        y0 = model(x0_var)
        y1 = model(x1_var)
        loss = criterion(y0, y1, target_var)
        losses.update(loss.data[0], x0_var.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), loss=losses))
    plotter.plot('loss', 'train', epoch, losses.avg)

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    parser.add_argument('--anno_txt')
    parser.add_argument('--name', default="ranknet")
    parser.add_argument('--ckpt', help='path to the checkpoint')

    args = parser.parse_args()
    main(args)

