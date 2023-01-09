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
from network.attension import ChannelAttention,SpatialAttention
from network.SelfAttension import Self_Attn


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
'''
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_size, out_size, 4, 2, 1)
        
        self.ca = ChannelAttention(out_size)
        self.sa = SpatialAttention()
        self.norm = nn.InstanceNorm2d(out_size)
        self.normalize = normalize
        self.dropout = dropout
        self.drop = nn.Dropout()
        #layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
        #if normalize:
            #layers.append(nn.InstanceNorm2d(out_size))
        
        #if dropout:
            #layers.append(nn.Dropout(dropout))
        #self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.leakyrelu(x)
        xc = self.conv(x)
        xa = self.ca(xc)*xc
        xa = self.sa(xa)*xa
        x = xc + xa
        if self.normalize:
            x = self.norm(x)
        if self.dropout:
            x = self.drop(x)
        
        return x
'''

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.LeakyReLU(0.2)]
        layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        
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
        x = torch.cat((x, skip_input), 1)

        return x

class SSAEncoder(nn.Module):
    """
    UNET ENCODER NETWORK
    """
    def __init__(self, isize=64, nz=100, nc=3, ndf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
        super(SSAEncoder, self).__init__()
        self.isize = isize
        self.down1 = UNetDown(nc, 64, normalize=False) #isize/2
        self.down2 = UNetDown(64, 128) #isize/4
        self.down3 = UNetDown(128,256) #isize/8
        self.down4 = UNetDown(256,512, dropout=0.5) #isize/16
        if isize==128:
            self.down5 = UNetDown(512,512, dropout=0.5) #isize/32
            if add_final_conv:
                self.conv1 = nn.Conv2d(512, nz, 4, 1, 0, bias=False) #isize/128
        elif isize==64:
            if add_final_conv:
                self.conv1 = nn.Conv2d(512, nz, 4, 1, 0, bias=False) #isize/64
        elif isize==32:
            if add_final_conv:
                self.conv1 = nn.Conv2d(256, nz, 4, 1, 0, bias=False) #isize/32
                
        main = nn.Sequential()
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(nc,  64),self.down1)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(64, 128),self.down2)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(128,256),self.down3)
        
        if isize==128:
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(256,512),self.down4)
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(512,512),self.down5)
            main.add_module('pyramid-conv-{0}-{1}'.format(512, nz),self.conv1)
        elif isize==64:
            main.add_module('pyramid-UNetDown-{0}-{1}'.format(256,512),self.down4)
            main.add_module('pyramid-conv-{0}-{1}'.format(512, nz),self.conv1)
        elif isize==32:
            main.add_module('pyramid-conv-{0}-{1}'.format(256, nz),self.conv1)
        
        self.main = main
        
        self.layers = list(main.children())
        
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        
        self.ca4 = ChannelAttention(512)
        self.sa4 = SpatialAttention()
        
        
        self.ca5 = ChannelAttention(512)
        self.sa5 = SpatialAttention()
        
    def forward(self, input):
        #print('\ninput shape:{}'.format(input.shape))
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        if self.isize==128:
            d5 = self.down5(d4)
            
            d5 = self.ca4(d5) * d5
            d5 = self.sa4(d5) * d5
        #d6 = self.down6(d5)
        
        d1 = self.ca1(d1) * d1
        d1 = self.sa1(d1) * d1
        
        d2 = self.ca2(d2) * d2
        d2 = self.sa2(d2) * d2
        
        d3 = self.ca3(d3) * d3
        d3 = self.sa3(d3) * d3
        
        d4 = self.ca4(d4) * d4
        d4 = self.sa4(d4) * d4
        
        
        
        
        output = self.main(input)
        #print('\output.shape:{}'.format(output.shape))
        if self.isize==128:
            d = [d1,d2,d3,d4,d5]
        elif self.isize==64:
            d = [d1,d2,d3,d4]
        elif self.isize==32:
            d = [d1,d2,d3]
        return d,output
    
    
class SSADecoder(nn.Module):
    """
    UNET DECODER NETWORK
    """
    def __init__(self,isize=64, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
        super(SSADecoder, self).__init__()
        
        self.isize = isize
        if isize==128:
            self.con1 = nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False)
        
            self.up1 = UNetUp(512, 256, dropout=0.5) # self.down4 = UNetDown(256,512, dropout=0.5)
            self.up2 = UNetUp(768, 256, dropout=0.5) #self.down3  = UNetDown(128, 256, dropout=0.5)
            self.up3 = UNetUp(512, 128, dropout=0.5) # self.down2 = UNetDown(64, 128)
            self.up4 = UNetUp(256,  64, dropout=0.5) # self.down1 = UNetDown(nc, 64, normalize=False)
        elif isize==64:
            self.con1 = nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False)
    
            self.up1 = UNetUp(512, 256, dropout=0.5) #self.down3 = UNetDown(128, 256, dropout=0.5)
            self.up2 = UNetUp(512, 128, dropout=0.5) # self.down2 = UNetDown(64, 128)
            self.up3 = UNetUp(256,  64, dropout=0.5) # self.down1 = UNetDown(nc, 64, normalize=False)
        elif isize==32:
            self.con1 = nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False)
            self.up1 = UNetUp(256, 128, dropout=0.0) # self.down2 = UNetDown(64, 128)
            self.up2 = UNetUp(256,  64, dropout=0.0) # self.down1 = UNetDown(nc, 64, normalize=False)
               
        '''
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),nn.Conv2d(128, nc, 3, padding=1), nn.Tanh()
        )
        '''
        self.final_feature = nn.Sequential(
            nn.Upsample(scale_factor=2),nn.Conv2d(128, nc, 3, padding=1)
        )
        
        self.tanh = nn.Tanh()
        self.attn1 = Self_Attn( 3, 'relu')
        #self.attn2 = Self_Attn( 128, 'relu')
        #self.attn3 = Self_Attn( 256, 'relu')
        #self.attn4 = Self_Attn( 512, 'relu')
        
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
            final_fea = self.final_feature(u2)
            final = self.tanh(final_fea)
            final_attn = self.attn1(final_fea)
            #print('final_attn {}'.format(final_attn.shape))
            
        return final,final_attn


class SSANetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, isize=64, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, extralayers=0):
        super(SSANetG, self).__init__()
        self.isize = isize
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.ngpu = ngpu
        self.extralayers = extralayers
        
        self.encoder1 = SSAEncoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.decoder = SSADecoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.encoder2 = SSAEncoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)

    def forward(self, x):
        d, latent_i = self.encoder1(x)
        gen_imag, gen_attn = self.decoder(latent_i,d)
        d, latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o, gen_attn
    

class SSANetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,isize=64, nc=3):
        super(SSANetD, self).__init__()
        #model = SSAEncoder(isize=isize, nz=1, nc=nc, ndf=64, ngpu=1, n_extra_layers=0)
        model_fusion = SSAEncoder(isize=isize, nz=1, nc=nc, ndf=64, ngpu=1, n_extra_layers=0)
        #self.concat = nn.cat(dim=0)
        #layers = list(model.main.children())
        layers = list(model_fusion.main.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())
        #self.input_attn = torch.ones(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
    def forward(self, x, attention):
        fusion = torch.cat((x,attention),1)
        features = self.features(fusion)
        #features = self.features(x)
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