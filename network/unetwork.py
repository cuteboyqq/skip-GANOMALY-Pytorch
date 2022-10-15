# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:43:52 2022

@author: User
"""

import torch.nn as nn


##
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

'''
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize=64, nz=100, nc=3, ndf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize=64, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """
    def __init__(self, isize=64, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, extralayers=0):
        super(NetG, self).__init__()
        self.isize = isize
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.ngpu = ngpu
        self.extralayers = extralayers
        self.encoder1 = Encoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.decoder = Decoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)
        self.encoder2 = Encoder(self.isize, self.nz, self.nc, self.ngf, self.ngpu, self.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o
    
    
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self,isize=64, nc=3):
        super(NetD, self).__init__()
        model = Encoder(isize=isize, nz=1, nc=nc, ndf=64, ngpu=1, n_extra_layers=0)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
'''
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
        x = torch.cat((x, skip_input), 1)

        return x

class UEncoder(nn.Module):
    """
    UNET ENCODER NETWORK
    """
    def __init__(self, isize=64, nz=100, nc=3, ndf=64, ngpu=1, n_extra_layers=0, add_final_conv=True):
        super(UEncoder, self).__init__()
        
        self.down1 = UNetDown(nc, 32, normalize=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128, dropout=0.5)
        self.down4 = UNetDown(128,256, dropout=0.5)
        #self.down5 = UNetDown(512, 512, dropout=0.5)
        #self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)
        
        if add_final_conv:
            self.conv1 = nn.Conv2d(256, nz, 4, 1, 0, bias=False)
            
        main = nn.Sequential()
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(nc,  32),self.down1)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(32,  64),self.down2)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(64, 128),self.down3)
        main.add_module('pyramid-UNetDown-{0}-{1}'.format(128,256),self.down4)
        #main.add_module('pyramid-UNetDown-{0}-{1}'.format(512, 512),self.down5)
        #main.add_module('pyramid-UNetDown-{0}-{1}'.format(512, 512),self.down6)
        main.add_module('pyramid-conv-{0}-{1}'.format(256, nz),self.conv1)
        
        self.main = main
        
    def forward(self, input):
        #print('\ninput shape:{}'.format(input.shape))
        d1 = self.down1(input)
        #print('\nd1.shape:{}'.format(d1.shape))
        d2 = self.down2(d1)
        #print('\nd2.shape:{}'.format(d2.shape))
        d3 = self.down3(d2)
        #print('\nd3.shape:{}'.format(d3.shape))
        d4 = self.down4(d3)
        #print('\nd4.shape:{}'.format(d4.shape))
        #d5 = self.down5(d4)
        #d6 = self.down6(d5)
        
        output = self.main(input)
        #print('\output.shape:{}'.format(output.shape))
        d = [d1,d2,d3,d4]
        return d,output
    
    
class UDecoder(nn.Module):
    """
    UNET DECODER NETWORK
    """
    def __init__(self,isize=64, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0):
        super(UDecoder, self).__init__()
        
        
        self.con1 = nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False)
        
        #self.up1 = UNetUp(512, 512, dropout=0.5)
        #self.up2 = UNetUp(1024, 512, dropout=0.5)
        #self.up3 = UNetUp(1024, 256, dropout=0.5)
        #self.up4 = UNetUp(512, 128)
        #self.up5 = UNetUp(256, 64)
        
        
        self.up1 = UNetUp(256, 128, dropout=0.5)
        self.up2 = UNetUp(256,  64, dropout=0.5)
        self.up3 = UNetUp(128,  32, dropout=0.5)
        
        #self.final = nn.Sequential(
        #    nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(64, nc, 4, padding=1), nn.Tanh()
        #)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),nn.Conv2d(64, nc, 3, padding=1), nn.Tanh()
        )
    def forward(self, input,d):
        
        #print('input:{}'.format(input.shape))
        c1 = self.con1(input)
        '''
        u1 = self.up1(c1, d[4])
        u2 = self.up2(u1, d[3])
        u3 = self.up3(u2, d[2])
        u4 = self.up4(u3, d[1])
        u5 = self.up5(u4, d[0])
        '''
        #print('c1: {}'.format(c1.shape))
        #print('d[2]: {}'.format(d[2].shape))
        u1 = self.up1(c1, d[2])
        #print('u1: {}'.format(u1.shape))
        #print('d[1]: {}'.format(d[1].shape))
        u2 = self.up2(u1, d[1])
        #print('d[0]: {}'.format(d[0].shape))
        #print('u2 :{}'.format(u2.shape))
        u3 = self.up3(u2, d[0])
        #print('u3 :{}'.format(u3.shape))
        final = self.final(u3)
        #print('final :{}'.format(final.shape))
        return self.final(u3)


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

    def __init__(self,isize=64, nc=3):
        super(UNetD, self).__init__()
        model = UEncoder(isize=isize, nz=1, nc=nc, ndf=64, ngpu=1, n_extra_layers=0)
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