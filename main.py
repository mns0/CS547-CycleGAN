from __future__ import print_function
from dataloader import get_data, Sample_from_Pool, cycle_data
#from train import train
#from test import test
#from models import Encoder, ResnetBlock, Decoder, Generator, Discriminator
from models import Encoder_cc, ResBlock_cc, Decoder_cc, generator_cc, discriminator_cc
import argparse
import os
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
#from torchsummary import summary
from train import train_step

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

print("torch.__version__: {}".format( torch.__version__ ))

#### Parse options #####
parser = argparse.ArgumentParser()
parser.add_argument('--datafolder', type=str, default="datasets", required=True, help='path to data')
parser.add_argument('--dataset',   type=str, default="monet2photo", required=True, help='dataset name')
parser.add_argument('--name',       type=str, default="tmp", required=True, help='Name of the run')
parser.add_argument('--mode',       type=str, default="train", required=True, help='(train or test)')
parser.add_argument('--res',       type=int, default=32, help='image height and width')
parser.add_argument('--crop_size',  type=str, default=128, help='size of the cropped image')
parser.add_argument('--batch_size',  type=int, default=32, help='size of the batch')
parser.add_argument('--num_epochs',  type=int, default=32, help='num of epochs')
parser.add_argument('--identity_loss',  type=float, default=0.5, help='Identity loss constant, 0 for no identity loss')
parser.add_argument('--lambda_identity_x',  type=float, default=10, help='Identity loss constant for cycle loss x > y > x')
parser.add_argument('--lambda_identity_y',  type=float, default=10, help='Identity loss constant for cycle loss y > x > y')
parser.add_argument('--save_dir',  type=str, help='Directory to save model - typically ./ckpts/{number}')
parser.add_argument('--wgan_lambda',  type=float, default=0, help='multiplier for WGAN loss (if 0, no WGAN loss, default, 10 is a typical value)')
#parser.add_argument('--two_step',  type=bool, default=False, help='two-step advasarial loss')


#########################

opt = parser.parse_args()

num_epochs = opt.num_epochs
batch_size = opt.batch_size

#for the identity loss
lambda_identity_loss = opt.identity_loss
lambda_idt_y = opt.lambda_identity_x
lambda_idt_x = opt.lambda_identity_y

wgan_lambda = opt.wgan_lambda
#saving the loss file
save_dir = opt.save_dir
save_log_file = save_dir + "/loss.log"

fh = open(save_log_file,'ab')

if opt.mode == "train":
    data_options = {'folder' : opt.datafolder, 'dataset': opt.dataset,
            'name' : opt.name, 'mode': opt.mode, 'res': opt.res, 'crop_size': opt.crop_size, 'batch_size': opt.batch_size}
    #trainloader = get_data(**data_options)

    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),])

    print("getting data from", opt.datafolder + '/' + opt.dataset)
    trainset = cycle_data(opt.datafolder + '/' + opt.dataset+'/')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    G_x2y = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    G_y2x = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    D_x, D_y =  discriminator_cc().cuda(), discriminator_cc().cuda()

    optimizer_G_x2y = torch.optim.Adam(G_x2y.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_G_y2x = torch.optim.Adam(G_y2x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_x = torch.optim.Adam(D_x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_y = torch.optim.Adam(D_y.parameters(), lr=0.0001, betas=(0,0.9))


    criterion_image = nn.L1Loss()
    criterion_type = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    x_fake_sample = Sample_from_Pool()
    y_fake_sample = Sample_from_Pool()

    for epoch in range(num_epochs):
        train_step(epoch,trainloader, G_x2y, G_y2x,D_x,D_y,\
               optimizer_G_x2y,optimizer_G_y2x,optimizer_D_x, optimizer_D_y,\
               criterion_image,criterion_type,criterion_identity, batch_size,\
               x_fake_sample,y_fake_sample,lambda_identity_loss,\
               lambda_idt_x,lambda_idt_y, wgan_lambda, save_dir,save_log_file,fh)

    fh.close()

else:
    #Potential validation code
    pass

