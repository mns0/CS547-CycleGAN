import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.distributed as dist


def train_step(G_x2y, G_y2x,D_x,D_y, optimizer_G_x2y,optimizer_G_y2x,optimizer_D_x, optimizer_D_y,criterion_image,criterion_type, X_real,Y_real, batch_size, x_fake_sample, y_fake_sample):

    G_x2y.train()
    G_y2x.train()
    D_x.train()
    D_y.train()


    G_x2y.zero_grad()
    G_y2x.zero_grad()
    X_real, Y_real = Variable(X_real.float()).cuda(), Variable(Y_real.float()).cuda()

    X_fake = G_y2x(Y_real)
    Y_fake = G_x2y(X_real)

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


    #for bluewaters
    for sing_opt in [optimizer_G_x2y, optimizer_G_y2x, optimizer_D_x, optimizer_D_y]:
        for group in sing_opt.param_groups:
            for p in group['params']:
                state = sing_opt.state[p]
                if(state['step']>=1024):
                    state['step'] = 1000



    return G_loss.item(), D_X_loss.item(), D_Y_loss.item()
