from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import model
from dataset import TextDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    required=True,
    default='folder',
    help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument(
    '--dataroot', required=True, default='./data/coco', help='path to dataset')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--imageSize',
    type=int,
    default=64,
    help='the height / width of the input image to network')
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
parser.add_argument(
    '--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument(
    '--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument(
    '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument(
    '--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument(
    '--netG', default='', help="path to netG (to continue training)")
parser.add_argument(
    '--netD', default='', help="path to netD (to continue training)")
parser.add_argument(
    '--outf',
    default='./output/',
    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print(
        "WARNING: You have a CUDA device, so you should probably run with --cuda"
    )

# if opt.dataset in ['imagenet', 'folder', 'lfw']:
#     # folder dataset
#     dataset = dset.ImageFolder(
#         root=opt.dataroot,
#         transform=transforms.Compose([
#             transforms.Scale(opt.imageSize),
#             transforms.CenterCrop(opt.imageSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))
# elif opt.dataset == 'lsun':
#     dataset = dset.LSUN(
#         db_path=opt.dataroot,
#         classes=['bedroom_train'],
#         transform=transforms.Compose([
#             transforms.Scale(opt.imageSize),
#             transforms.CenterCrop(opt.imageSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))
# elif opt.dataset == 'cifar10':
#     dataset = dset.CIFAR10(
#         root=opt.dataroot,
#         download=True,
#         transform=transforms.Compose([
#             transforms.Scale(opt.imageSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))
# elif opt.dataset == 'fake':
#     dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
#                             transform=transforms.ToTensor())
# assert dataset

image_transform = transforms.Compose([
    transforms.RandomCrop(opt.imageSize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = TextDataset(opt.dataroot, transform=image_transform)

## TODO: Make a new DataLoader and Dataset to include embeddings
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = model._netG(ngpu, nz, ngf, nc, nte, nt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model._netD(ngpu, nc, ndf, nte, nt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

## TODO: Change the error loss function to include embeddings [refer main_cls.lua on the original paper repo]

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, txt_embedding = data  # Text embedding
        batch_size = real_cpu.size(0)
        txt_embedding = Variable(txt_embedding)

        if opt.cuda:
            real_cpu = real_cpu.cuda()
            txt_embedding = txt_embedding.cuda()

        ## TODO: Generate fake images first

        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(
            label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print(
            '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader), errD.data[0],
               errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf, normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
