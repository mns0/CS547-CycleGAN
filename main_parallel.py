from dataloader import get_data, Sample_from_Pool
#from models import Encoder, ResnetBlock, Decoder, Generator, Discriminator
from models_parallel import Encoder_cc, ResBlock_cc, Decoder_cc, generator_cc, discriminator_cc, Discriminator
import argparse
import os
import torch
import torch.nn as nn
import time
from torch.autograd import Variable

import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI


import pycuda
from pycuda import compiler
import pycuda.driver as drv
from train import train_step

from torch.nn.parallel import DistributedDataParallel


cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor

drv.init()
for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print (ordinal, dev.name())




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


cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')

print("c0", cuda0)
print("c1", cuda1)


if opt.mode == "train":
    data_options = {'folder' : opt.datafolder, 'dataset': opt.dataset,
            'name' : opt.name, 'mode': opt.mode, 'res': opt.res, 'crop_size': opt.crop_size, 'batch_size': opt.batch_size}
    trainloader = get_data(**data_options)

    print("WORLD SIZE", torch.distributed.get_world_size())
    G_x2y = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    G_y2x = generator_cc(Encoder_cc, ResBlock_cc, Decoder_cc).cuda()
    D_x =  discriminator_cc().cuda()
    D_y =  discriminator_cc().cuda()
    #D_x, D_y =  Discriminator().cuda(), Discriminator().cuda()

    pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))

    print("rank", comm.Get_rank())

    pg_G_x2y = nn.DataParallel(G_x2y, device_ids=[0] )
    pg_G_y2x = nn.DataParallel(G_y2x, device_ids=[0,1] )
    print("Process groups", pg_G_x2y, pg_G_y2x)


    optimizer_G_x2y = torch.optim.Adam(G_x2y.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_G_y2x = torch.optim.Adam(G_y2x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_x = torch.optim.Adam(D_x.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_D_y = torch.optim.Adam(D_y.parameters(), lr=0.0001, betas=(0,0.9))

    criterion_image = nn.L1Loss()
    criterion_type = nn.L1Loss()


    x_fake_sample = Sample_from_Pool()
    y_fake_sample = Sample_from_Pool()

    #for epoch in range(num_epochs):
    #    print("runnning epoch", epoch)
    #    start_time = time.time()

    #    for batch_idx, (X_real, Y_real) in enumerate(trainloader):

    #        if Y_real.shape[0] < batch_size:
    #            continue;

    #        gloss, dxloss, dyloss = train_step(G_x2y, G_y2x,D_x,D_y, optimizer_G_x2y,optimizer_G_y2x,optimizer_D_x, optimizer_D_y,criterion_image,criterion_type, X_real,Y_real, batch_size, x_fake_sample,y_fake_sample)
    #        #print(gloss,dxloss,dyloss)
    #    print('----EPOCH{} FINISHED'.format(epoch))

else:
    #TODO: Validation code
    pass

