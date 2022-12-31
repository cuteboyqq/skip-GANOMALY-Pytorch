#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:59:04 2022

@author: ali
"""

from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from network.unetwork import UNetG, UNetD, weights_init
from network.loss import l2_loss

from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class UGanomaly(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'skip-Ganomaly'
   
    def __init__(self,args):
                 
        super(UGanomaly, self).__init__()
        
        self.batchsize = args.batch_size
        self.test_batchsize = args.test_batchsize
        self.isize = args.img_size
        self.lr = args.lr
        self.beta1 = 0.5
        self.isTrain = args.train
        self.resume = args.weights
        self.test_model = args.save_dir
        self.nz = args.nz
        self.nc = args.nc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.save_test_images = True
        

        ##
        # Create and initialize networks.
        self.netg = UNetG(self.isize,self.nc,self.nz).to(self.device)
        self.netd = UNetD(self.isize,self.nc).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.resume != '':
            print("\nLoading pre-trained networks.")
            #self.iter = torch.load(os.path.join(self.resume, 'netG.pt'))['epoch']
            #self.netg.load_state_dict(torch.load(os.path.join(self.resume, 'netG.pt'))['state_dict'])
            #self.netd.load_state_dict(torch.load(os.path.join(self.resume, 'netD.pt'))['state_dict'])
            self.netg.load_state_dict(torch.load(os.path.join(self.resume, 'netG.pt')))
            self.netd.load_state_dict(torch.load(os.path.join(self.resume, 'netD.pt')))
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(self.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.isTrain:
            self.netg.train()
            self.netd.train()
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    ##
    def forward_g(self,x):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(x)

    ##
    def forward_d(self,x):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(x)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())
    
    
    ##
    def backward_g(self,x):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(x)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, x)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * 1 + \
                     self.err_g_con * 50 + \
                     self.err_g_enc * 1
        self.err_g.backward(retain_graph=True)
        
        return self.err_g

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        #print(len(self.pred_real))
        #print(len(self.real_label))
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
        
        return self.err_d

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')
    ##
    def forward(self,x):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g(x)
        self.forward_d(x)
        
        # Backward-pass
        # netg
        #if self.isTrain:
        self.optimizer_g.zero_grad()
        error_g = self.backward_g(x)
        if self.isTrain:
            self.optimizer_g.step()

        # netd
        #if self.isTrain:
        self.optimizer_d.zero_grad()
        error_d = self.backward_d()
        if self.isTrain:
            self.optimizer_d.step()
        #if self.err_d.item() < 1e-5: self.reinit_d()
        
        return error_g, error_d, self.fake, self.netg, self.netd #error_d
    
    def load_weights(self, epoch=None, is_best:bool=False, path=None):
        """ Load pre-trained weights of NetG and NetD
        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})
        Raises:
            Exception -- [description]
            IOError -- [description]
        """
        
        #if epoch is None and is_best is False:
            #raise Exception('Please provide epoch to be loaded or choose the best epoch.')
        '''
        if is_best:
            fname_g = f"netG_best.pth"
            fname_d = f"netD_best.pth"
        else:
            fname_g = f"netG_{epoch}.pth"
            fname_d = f"netD_{epoch}.pth"

        if path is None:
            path_g = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_g}"
            path_d = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_d}"

        # Load the weights of netg and netd.
        print('>> Loading weights...')
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        '''
        try:
            #self.netg.load_state_dict(weights_g)
            #self.netd.load_state_dict(weights_d)
            
            self.netg.load_state_dict(torch.load(os.path.join(self.test_model, 'netG.pt')))
            self.netd.load_state_dict(torch.load(os.path.join(self.test_model, 'netD.pt')))
            
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')



    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth
        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
                
    
    ##
    def get_current_images(self):
        """ Returns current images.
        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed
    
    ##
    def test(self, test_loader, plot_hist=False):
        """ Test GANomaly model.
        Args:
            data ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            #if self.load_weights:
            self.load_weights(is_best=True)

            self.phase = 'test'

            scores = {}
            
            self.test_dataset = test_loader
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.test_dataset)*self.test_batchsize,), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.test_dataset)*self.test_batchsize,), dtype=torch.long, device=self.device)
            self.features  = torch.zeros(size=(len(self.test_dataset)*self.test_batchsize, self.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            pbar = tqdm(test_loader)
            for i, data in enumerate(test_loader, 0):
                self.total_steps += self.test_batchsize
                epoch_iter += self.test_batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                #print('self.input : {}'.format(self.input.shape))
                #print('data: {}'.format(data.shape))
                self.fake, self.latent_i, self.latent_o = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat
                #print('error : {}'.format(error.shape))
                time_o = time.time()

                self.an_scores[i*self.test_batchsize: i*self.test_batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.test_batchsize: i*self.test_batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)
                
                self.outf = './runs/detect/'
                
                # Save test images.
                if self.save_test_images:
                    dst = os.path.join(self.outf, '2022-12-31', 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                plt.ion()
                # Create data frame for scores and labels.
                scores['scores'] = self.an_scores
                scores['labels'] = self.gt_labels
                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv("histogram.csv")

                # Filter normal and abnormal scores.
                abn_scr = hist.loc[hist.labels == 1]['scores']
                nrm_scr = hist.loc[hist.labels == 0]['scores']

                # Create figure and plot the distribution.
                # fig, ax = plt.subplots(figsize=(4,4));
                sns.distplot(nrm_scr, label=r'Normal Scores')
                sns.distplot(abn_scr, label=r'Abnormal Scores')

                plt.legend()
                plt.yticks([])
                plt.xlabel(r'Anomaly Scores')

            ##
            # PLOT PERFORMANCE
            #if self.opt.display_id > 0 and self.opt.phase == 'test':
            #counter_ratio = float(epoch_iter) / len(self.test_dataset)
            #self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            ##
            # RETURN
            return performance,auc
