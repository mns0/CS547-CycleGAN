import torch
import torch.nn as nn


#A simple patchgan discriminator
class Generator(nn.Module):
    def __init__(self, Encoder, ResnetBlock, Decoder, num_of_resblocks=3):
        super(Generator, self).__init__()
        self.Encoder = Encoder
        self.Resnet = ResnetBlock
        self.Decoder = Decoder
        self.num_of_resblocks = num_of_resblocks

    def forward(self,x):
        x = self.Encoder(x)
        for i in range(self.num_of_resblocks):
            x = self.Resnet(x) 
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
                nn.ReflectionPad2d(40),
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9,
                    padding=4,stride=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1,stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True))

        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1,stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True))

    def forward(self,x):
        x  = self.conv1(x)
        x  = self.conv2(x)
        x  = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, padding=0,stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, padding=0,stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True))

        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 3, 5, padding=0,stride=1),
                nn.BatchNorm2d(3),
                nn.LeakyReLU(inplace=True))

    def forward(self,x):
        x  = self.conv1(x)
        x  = self.conv2(x)
        x  = self.conv3(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(128, 128,3, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 128,3, padding=0),
                nn.BatchNorm2d(128))

    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        #center crop the tensor
        res = res.narrow(2,2,res.shape[2]-4)
        res = res.narrow(3,2,res.shape[3]-4)
        x+=res
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, True))

        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, True))

        self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True))

        self.conv5 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, True))

        self.fc = nn.Linear(512,1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(4,4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1,512)
        x = self.fc(x)
        return x




class Encoder_cc(nn.Module):
    def __init__(self):
        super(Encoder_cc, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.lru = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = self.lru(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.lru(self.norm2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.lru(self.norm3(self.conv3(x)))
        x = self.maxpool(x)

        return x


class ResBlock_cc(nn.Module):
    def __init__(self):
        super(ResBlock_cc, self).__init__()
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.lru = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm2d(128)

    def forward(self,x):

        temp_in = x
        x = self.lru(self.norm3(self.conv4(x)))
        x = self.norm3(self.conv4(x))

        return x + temp_in

class Decoder_cc(nn.Module):
    def __init__(self):
        super(Decoder_cc, self).__init__()
        self.conv5 = nn.ConvTranspose2d(128, 64, 4,padding=1,stride = 2)
        self.conv6 = nn.ConvTranspose2d(64, 32, 4,padding=1,stride = 2)
        self.conv7 = nn.ConvTranspose2d(32, 3, 4,padding=1,stride = 2)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm1 = nn.BatchNorm2d(32)
        self.lru = nn.LeakyReLU()

    def forward(self,x):
        x = self.lru(self.norm2(self.conv5(x)))
        x = self.lru(self.norm1(self.conv6(x)))
        x = self.conv7(x)
        return x

class generator_cc(nn.Module):
    def __init__(self,Encoder_cc,ResBlock_cc,Decoder_cc):
        super(generator_cc, self).__init__()
        self.encoder = Encoder_cc()
        self.resblock = ResBlock_cc()
        self.decoder = Decoder_cc()

    def forward(self, x):
        x = self.encoder(x)
        for i in range(3):
            x = self.resblock(x)
        x = self.decoder(x)
        return x

class discriminator_cc(nn.Module):
    def __init__(self):
        super(discriminator_cc, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding = 1)#128
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)#64
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding = 1)#16
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3,padding = 1)#4
        self.norm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,512,3,padding = 1)#1
        self.norm5 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(4,4)
        self.maxpool3 = nn.MaxPool2d(3,3)
        self.fc = nn.Linear(512,1)

    def forward(self,x):
        #print("1", x.size())
        x = self.relu(self.norm1(self.conv1(x)))
        #print("2", x.size())
        x = self.maxpool1(x)
        #print("3", x.size())
        x = self.relu(self.norm2(self.conv2(x)))
        #print("4", x.size())
        x = self.maxpool1(x)
        #print("5", x.size())
        x = self.relu(self.norm3(self.conv3(x)))
        #print("6", x.size())
        x = self.maxpool2(x)
        #print("7", x.size())
        x = self.relu(self.norm4(self.conv4(x)))
        #print("8", x.size())
        x = self.maxpool2(x)
        #print("9", x.size())
        x = self.relu(self.norm5(self.conv5(x)))
        #print("10", x.size())
        x = self.maxpool1(x)
        #print("11", x.size())
        x = x.view(-1,512)
        #print("12", x.size())
        x = self.fc(x)
        #print("13", x.size())
        return x
