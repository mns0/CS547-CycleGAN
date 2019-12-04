from dataloader import get_data, Sample_from_Pool
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
from torchsummary import summary

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
#########################

opt = parser.parse_args()

num_epochs = opt.num_epochs
batch_size = opt.batch_size

if opt.mode == "train":
    data_options = {'folder' : opt.datafolder, 'dataset': opt.dataset,
            'name' : opt.name, 'mode': opt.mode, 'res': opt.res, 'crop_size': opt.crop_size, 'batch_size': opt.batch_size}
    trainloader = get_data(**data_options)

    #G_x2y = Generator(Encoder(), ResnetBlock(), Decoder(), 5).cuda()
    #G_y2x = Generator(Encoder(), ResnetBlock(), Decoder(), 5).cuda()
    #D_x, D_y =  Discriminator().cuda(), Discriminator().cuda()

    G_x2y = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    G_y2x = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    D_x, D_y =  discriminator_cc().cuda(), discriminator_cc().cuda()


    optimizer_G_x2y = torch.optim.Adam(G_x2y.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_G_y2x = torch.optim.Adam(G_y2x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_x = torch.optim.Adam(D_x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_y = torch.optim.Adam(D_y.parameters(), lr=0.0001, betas=(0,0.9))

    criterion_image = nn.L1Loss()
    criterion_type = nn.L1Loss()


    x_fake_sample = Sample_from_Pool()
    y_fake_sample = Sample_from_Pool()

    for epoch in range(num_epochs):

        start_time = time.time()

        for batch_idx, (X_real, Y_real) in enumerate(trainloader):

            G_x2y.train()
            G_y2x.train()
            D_x.train()
            D_y.train()

            if(Y_real.shape[0] < batch_size):
                continue

            G_x2y.zero_grad()
            G_y2x.zero_grad()
            #print(type(Y_real))
            X_real, Y_real = Variable(X_real.float()).cuda(), Variable(Y_real.float()).cuda()
            #print(type(Y_real))

            X_fake = G_y2x(Y_real)
            Y_fake = G_x2y(X_real)

            #print(Y_real.size(),X_fake.size())
            #summary(G_y2x,(3,256,256))
            X_cycle = G_y2x(Y_fake)
            Y_cycle = G_x2y(X_fake)

            D_X_real = D_x(X_real)
            D_Y_real = D_y(Y_real)
            D_X_fake = D_x(X_fake)
            D_Y_fake = D_y(Y_fake)

            loss_cycle = criterion_image(X_real,X_cycle)+criterion_image(Y_real,Y_cycle)
            real_label = Variable(torch.ones(D_Y_fake.size())).cuda()
            loss_G_X2Y = criterion_type(D_Y_fake,real_label)
            loss_G_Y2X = criterion_type(D_X_fake,real_label)

            G_loss = loss_G_X2Y + loss_G_Y2X + 10*loss_cycle
            G_loss.backward()

            optimizer_G_x2y.step()
            optimizer_G_y2x.step()

            X_fake = Variable(torch.Tensor(x_fake_sample([X_fake.cpu().data.numpy()])[0])).cuda()
            Y_fake = Variable(torch.Tensor(y_fake_sample([Y_fake.cpu().data.numpy()])[0])).cuda()


            D_x.zero_grad()
            D_y.zero_grad()
            D_X_real = D_x(X_real)
            D_Y_real = D_y(Y_real)
            D_X_fake = D_x(X_fake)
            D_Y_fake = D_y(Y_fake)
            real_label = Variable(torch.ones(D_Y_fake.size())).cuda()
            fake_label = Variable(torch.zeros(D_Y_fake.size())).cuda()

            D_X_loss = criterion_type(D_X_fake,fake_label)+criterion_type(D_X_real,real_label)
            D_Y_loss = criterion_type(D_Y_fake,fake_label)+criterion_type(D_Y_real,real_label)

            D_X_loss.backward()
            optimizer_D_x.step()

            D_Y_loss.backward()
            optimizer_D_y.step()

            print(G_loss.item(),D_X_loss.item(),D_Y_loss.item())

        print('----EPOCH{} FINISHED'.format(epoch))


else:
    #TODO: Validation code
    pass

