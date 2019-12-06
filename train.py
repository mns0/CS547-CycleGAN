import torch
import torch.nn as nn
import time
from torch.autograd import Variable, grad
import torch.distributed as dist

def calc_gradient_penalty(netD, real_data, fake_data, wgan_lambda, batch_size):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, real_data.shape[2], real_data.shape[3])
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, fake_data.shape[2], fake_data.shape[3])
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates  = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * wgan_lambda

    return gradient_penalty


def train_step(epoch,trainloader, G_x2y, G_y2x, D_x, D_y, optimizer_G_x2y,optimizer_G_y2x,optimizer_D_x, optimizer_D_y, criterion_image, criterion_type, criterion_identity , batch_size, x_fake_sample,y_fake_sample,lambda_identity_loss, lambda_idt_x,lambda_idt_y, wgan_lambda, save_dir,save_log_file):

    start_time = time.time()
    for batch_idx, (X_real, Y_real) in enumerate(trainloader):

        if(Y_real.shape[0] < batch_size):
            continue

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

        #identity loss
        loss_idt_A = 0
        loss_idt_B = 0
        #values taken from cyclegan defaults
        #lambda_identity_loss = 0.5
        #lambda_y = 10.0
        #lambda_x = 10.0

        if lambda_identity_loss > 0:
            I_x = G_x2y(Y_real)
            I_y = G_y2x(X_real)

            loss_idt_A = criterion_identity(I_x,Y_real) * lambda_idt_y * lambda_identity_loss
            loss_idt_B = criterion_identity(I_y,X_real) * lambda_idt_x * lambda_identity_loss

        G_loss = loss_G_X2Y + loss_G_Y2X + 10*loss_cycle + loss_idt_A + loss_idt_B
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

        wgan_loss_d_x = 0
        wgan_loss_d_y = 0

        if wgan_lambda > 0:
            wgan_loss_d_x = calc_gradient_penalty(D_x, Y_real, Y_fake, wgan_lambda, batch_size)
            wgan_loss_d_y = calc_gradient_penalty(D_y, X_real, X_fake, wgan_lambda, batch_size)

        D_X_loss = criterion_type(D_X_fake,fake_label)+criterion_type(D_X_real,real_label) + wgan_loss_d_x
        D_Y_loss = criterion_type(D_Y_fake,fake_label)+criterion_type(D_Y_real,real_label) + wgan_loss_d_y

        D_X_loss.backward()
        optimizer_D_x.step()

        D_Y_loss.backward()
        optimizer_D_y.step()

    #save models every 5 epochs

    if epoch % 5 == 0:
        torch.save(G_x2y.state_dict(), f'{save_dir}/G_x2y_epoch-{epoch}.ckpt')
        torch.save(G_y2x.state_dict(), f'{save_dir}/G_y2x_epoch-{epoch}.ckpt')
        torch.save(D_x.state_dict(),   f'{save_dir}/D_x_epoch-{epoch}.ckpt')
        torch.save(D_y.state_dict(),   f'{save_dir}/D_y_epoch-{epoch}.ckpt')

    #for bluewaters
    for sing_opt in [optimizer_G_x2y, optimizer_G_y2x, optimizer_D_x, optimizer_D_y]:
        for group in sing_opt.param_groups:
            for p in group['params']:
                state = sing_opt.state[p]
                if(state['step']>=1024):
                    state['step'] = 1000


    delta_t = time.time() - start_time

    print(epoch, G_loss.item(), D_X_loss.item(), D_Y_loss.item(), delta_t)
    sav_dat = [epoch, G_loss.item(), D_X_loss.item(), D_Y_loss.item(), delta_t]
    np.savetxt(fh,sav_dat)

    print('----EPOCH{} FINISHED'.format(epoch), )
   # return G_loss.item(), D_X_loss.item(), D_Y_loss.item()
