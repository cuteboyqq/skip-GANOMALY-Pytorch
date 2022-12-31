# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:43:52 2022

@author: User
"""
import torch.nn as nn

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

#==================================================Alister add 2022-10-14======================
import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        if skip_input==None:
            x = x
        else:
            x = torch.cat((x, skip_input), 1)
        return x

class UEncoder(nn.Module):
    """
    UNET ENCODER NETWORK
    """
    def __init__(self, isize=64, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
        super(UEncoder, self).__init__()
        self.isize = isize
        self.down1 = UNetDown(nc, ngf, normalize=False) #isize/2
        self.down2 = UNetDown(ngf, ngf*2) #isize/4
        self.down3 = UNetDown(ngf*2, ngf*4, dropout=0.5) #isize/8
        self.down4 = UNetDown(ngf*4,ngf*8, dropout=0.5) #isize/16
        if isize==128:
            self.down5 = UNetDown(ngf*8,ngf*8, dropout=0.5) #isize/32
            if add_final_conv:
                self.conv1 = nn.Conv2d(ngf*8, nz, 4, 1, 0, bias=False) #isize/128
        elif isize==64:
            if add_final_conv:
                self.conv1 = nn.Conv2d(ngf*8, nz, 4, 1, 0, bias=False) #isize/64
        elif isize==32:
            if add_final_conv:
                self.conv1 = nn.Conv2d(ngf*4, nz, 4, 1, 0, bias=False) #isize/32
                
        main = nn.Sequential()
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(nc,  ngf),self.down1)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(ngf,  ngf*2),self.down2)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(ngf*2, ngf*4),self.down3)
        
        if isize==128:
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(ngf*4,ngf*8),self.down4)
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(ngf*8,ngf*8),self.down5)
            main.add_module('pyramid-conv-{0}-{1}'.format(512, nz),self.conv1)
        elif isize==64:
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(ngf*4,ngf*8),self.down4)
            main.add_module('pyramid-conv-{0}-{1}'.format(ngf*8, nz),self.conv1)
        elif isize==32:
            main.add_module('pyramid-conv-{0}-{1}'.format(ngf*4, nz),self.conv1)
            
        self.main = main
        
        self.layers = list(main.children())
        
    def forward(self, input):
        #print('\ninput shape:{}'.format(input.shape))
        d1 = self.down1(input) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        
        if self.isize==128:
            d4 = self.down4(d3) #512
            d5 = self.down5(d4) #512
        elif self.isize==64:
            d4 = self.down4(d3) #512
        
        #d6 = self.down6(d5)
        
    
        
        
        output = self.main(input)
        #print('\output.shape:{}'.format(output.shape))
        if self.isize==128:
            d = [d1,d2,d3,d4,d5]
        elif self.isize==64:
            d = [d1,d2,d3,d4]
        elif self.isize==32:
            d = [d1,d2,d3]
        return d,output
    
    
class UDecoder(nn.Module):
    """
    UNET DECODER NETWORK
    """
    def __init__(self,isize=64, nz=100, nc=3, ndf=64, ngpu=1, n_extra_layers=0):
        super(UDecoder, self).__init__()
        
        self.isize = isize
        if isize==128:
            self.con1 = nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False)
        
            self.up1 = UNetUp(ndf*8, ndf*4, dropout=0.5) # self.down4 = UNetDown(128,256, dropout=0.5)
            self.up2 = UNetUp(ndf*12, ndf*4, dropout=0.5) #self.down3 = UNetDown(64, 128, dropout=0.5)
            self.up3 = UNetUp(ndf*8, ndf*2, dropout=0.5) # self.down2 = UNetDown(32, 64)
            self.up4 = UNetUp(ndf*4, ndf*2, dropout=0.5) # self.down1 = UNetDown(nc, 32, normalize=False)
        elif isize==64:
            self.con1 = nn.ConvTranspose2d(nz, ndf*8, 4, 1, 0, bias=False)
    
            self.up1 = UNetUp(ndf*8, ndf*4, dropout=0.5) #self.down3 = UNetDown(64, 128, dropout=0.5)
            self.up2 = UNetUp(ndf*8, ndf*2, dropout=0.5) # self.down2 = UNetDown(32, 64)
            self.up3 = UNetUp(ndf*4, ndf*2, dropout=0.5) # self.down1 = UNetDown(nc, 32, normalize=False)
        elif isize==32:
            self.con1 = nn.ConvTranspose2d(nz, ndf*4, 4, 1, 0, bias=False)
    
            self.up1 = UNetUp(ndf*4, ndf*2, dropout=0.75) #self.down3 = UNetDown(64, 128, dropout=0.5)
            self.up2 = UNetUp(ndf*4, 64, dropout=0.75) # self.down2 = UNetDown(32, 64)
            
               
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),nn.Conv2d(128, nc, 3, padding=1), nn.Tanh()
        )
    def forward(self, input,d):
        #print('input:{}'.format(input.shape))
        c1 = self.con1(input)
        if self.isize==128:
            u1 = self.up1(c1, d[3])
            u2 = self.up2(u1, d[2])
            u3 = self.up3(u2, d[1])
            u4 = self.up4(u3, d[0])
            final = self.final(u4)
        elif self.isize==64:
            u1 = self.up1(c1, d[2])
            u2 = self.up2(u1, d[1])
            u3 = self.up3(u2, d[0])
            final = self.final(u3)
        elif self.isize==32:
            u1 = self.up1(c1, d[1])
            u2 = self.up2(u1, d[0])
            final = self.final(u2)
        return final


class UNetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, isize=64, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, extralayers=0):
        super(UNetG, self).__init__()
        self.isize = isize
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.ngpu = ngpu
        self.extralayers = extralayers
        
        self.encoder1 = UEncoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.decoder = UDecoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.encoder2 = UEncoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)

    def forward(self, x):
        d, latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i,d)
        d, latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o
    

class UNetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,isize=64, nc=3, ndf=64):
        super(UNetD, self).__init__()
        model = UEncoder(isize=isize, nz=1, nc=nc, ngf=ndf, ngpu=1, n_extra_layers=0)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        #print('\nfeature : {}'.format(features.shape))
        classifier = self.classifier(features)
        #print('\nclassifier : {}'.format(classifier.shape))
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
    
# Code Unet Example from discogan model.py
'''
class GeneratorUNet(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorUNet, self).__init__()
        channels, _, _ = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
       
        #padding (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries. If a 4-tuple, uses
        #(\text{padding\_left}padding_left, \text{padding\_right}padding_right, \text{padding\_top}padding_top, \text{padding\_bottom}padding_bottom)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, channels, 4, padding=1), nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
    '''