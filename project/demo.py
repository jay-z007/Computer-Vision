from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import model
from dataset import TextDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataroot', required=True, default='./data/coco', help='path to dataset')
parser.add_argument(
    '--nte',
    type=int,
    default=1024,
    help='the size of the text embedding vector')
parser.add_argument(
    '--nt',
    type=int,
    default=256,
    help='the reduced size of the text embedding vector')
parser.add_argument(
    '--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument(
    '--netG', default='', help="path to netG (to continue training)")
opt = parser.parse_args()


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)  #use random.randint(1, 10000) for randomness, shouldnt be done when we want to continue training from a checkpoint
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True


image_transform = transforms.Compose([
    transforms.RandomCrop(opt.imageSize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1))
])

dataset = TextDataset(opt.dataroot, transform=image_transform)

## Completed - TODO: Make a new DataLoader and Dataset to include embeddings
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nt = int(opt.nt)
nte = int(opt.nte)


netG = model._netG(ngpu, nz, ngf, nc, nte, nt)