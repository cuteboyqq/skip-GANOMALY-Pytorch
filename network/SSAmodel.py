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

from network.SSAnetwork import SSANetG, SSANetD, weights_init
from network.loss import l2_loss
from network.SelfAttension import Self_Attn
import random



class SSAGanomaly(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'Skip-Attention-Ganomaly'
   
    def __init__(self,args):
                 
        super(SSAGanomaly, self).__init__()
        
        self.batchsize = args.batch_size
        self.isize = args.img_size
        self.lr = args.lr
        self.beta1 = 0.5
        self.isTrain = args.train
        self.resume = args.weights
        self.nz = args.nz
        self.nc = args.nc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        
        
        self.save_dir = args.save_dir
        
        
        #Alister 2023-01-09
        self.attn = Self_Attn( 1, 'relu').to(self.device)
        ##
        # Create and initialize networks.
        self.netg = SSANetG(self.isize,self.nc,self.nz).to(self.device)
        self.netd = SSANetD(self.isize,self.nc).to(self.device)
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
        #Alister 2023-01-08 add attetion loss
        self.l_attn = l2_loss
        ##
        # Initialize input tensors.
        #Alister 2023-01-08 add input attension ground truth 
        self.input_attn = torch.ones(size=(self.batchsize, self.isize*self.isize, self.isize*self.isize), dtype=torch.float32, device=self.device)
        
        self.input = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.input_2 = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        self.x2 = torch.empty(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
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
        self.fake, self.latent_i, self.latent_o, self.fake_attn = self.netg(x)
        self.x2.copy_(x)
        self.anomaly, self.anomaly_attn = self.cutout(self.x2)
        self.fake_anomaly, self.latent_i_anomaly, self.latent_o_anomaly, self.fake_anomaly_attn = self.netg(self.anomaly)

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
        self.err_g_ano_con = self.l_con(self.anomaly, self.fake_anomaly)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g_attn = self.l_attn(self.input_attn, self.fake_attn)
        self.err_g_ano_attn = self.l_attn(self.anomaly_attn, self.fake_anomaly_attn)
        self.err_g = self.err_g_adv * 1 + \
                     self.err_g_con * 50 - \
                     self.err_g_ano_con * 0 + \
                     self.err_g_enc * 1 + \
                     (self.err_g_attn + self.err_g_ano_attn) * 50
        
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
        #self.anomaly, self.anomaly_attn = self.cutout(x)
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
    
    ##
    def renormalize(self, tensor,minTo=0,maxTo=255):
            minFrom= tensor.min()
            maxFrom= tensor.max()
            minTo = minTo
            maxTo= maxTo
            return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    ##
    def cutout(self, im, p=1.0):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
        #print("im shape : {}".format(im.shape))
        #im_ano = torch.ones(size=(self.batchsize, 3, self.isize, self.isize), dtype=torch.float32, device=self.device)
        #im_ano.resize_(im[0].size()).copy_(im[0])
        im_mask = torch.ones(size=(self.batchsize, 1, self.isize, self.isize), dtype=torch.float32, device=self.device)
        #ano_attn = torch.ones(size=(self.batchsize, self.isize*self.isize, self.isize*self.isize), dtype=torch.float32, device=self.device)
        #im_attn = im_attn
        if random.random() < p:
            batch_size, c, h, w = im.shape[:]
            for i in range(batch_size):
                #print("h: {}, w: {}".format(h,w))
                scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
                for s in scales:
                    mask_h = random.randint(1, int(h * s))  # create random masks
                    mask_w = random.randint(1, int(w * s))
            
                    # box
                    xmin = max(0, random.randint(0, w) - mask_w // 2)
                    ymin = max(0, random.randint(0, h) - mask_h // 2)
                    xmax = min(w, xmin + mask_w)
                    ymax = min(h, ymin + mask_h)
            
                    # apply random color mask
                    #im_renorm = self.renormalize(im,0,255)
                    #im_renorm[i,:,ymin:ymax, xmin:xmax] = torch.tensor(random.randint(64, 191))
                    im[i,:,ymin:ymax, xmin:xmax] = torch.tensor(random.random())
                    #im = self.renormalize(im_renorm,0,1)
                    im_mask[i,:,ymin:ymax, xmin:xmax] =  torch.tensor(0.0)
                    
                    # return unobscured labels
                    #if len(labels) and s > 0.03:
                        #box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                        #ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                        #labels = labels[ioa < 0.60]  # remove >60% obscured labels
            im_attn = self.attn(im_mask) #error 2023-01-09
                    
        return im,im_attn
    
    ##
    def get_current_images(self):
        """ Returns current images.
        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        
        reals_ano = self.anomaly.data
        fakes_ano = self.fake_anomaly.data
        
        #fixed = self.netg(self.fixed_input)[0].data
        
        
        
        return reals, fakes, reals_ano, fakes_ano
    
    
    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth
        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.input_2.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
    
    ##
    def load_weights(self):
        # Load the weights of netg and netd.
        #print('>> Loading weights...')
        try:
            self.netg.load_state_dict(torch.load(os.path.join(self.save_dir, 'netG.pt')))
            self.netd.load_state_dict(torch.load(os.path.join(self.save_dir, 'netD.pt')))
        except IOError:
            raise IOError("netG weights not found")
        #print('   Done.')
    
    ##
    def test(self, test_data_loader):
        Skip_SelfAttention_Ganomaly = True
        """ Test GANomaly model.
        Args:
            data ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            
            self.load_weights()

            self.phase = 'test'
            
            self.test_dataset = test_data_loader
            scores = {}

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.test_dataset)*self.batchsize,), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.test_dataset)*self.batchsize,), dtype=torch.long, device=self.device)
            self.features  = torch.zeros(size=(len(self.test_dataset)*self.batchsize, self.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            pbar = tqdm(self.test_dataset,)
            for i, data in enumerate(pbar, 0):
                self.total_steps += self.batchsize
                epoch_iter += self.batchsize
                time_i = time.time()

                # Forward - Pass
                self.set_input(data)
                if Skip_SelfAttention_Ganomaly:
                    self.fake, self.latent_i, self.latent_o, self.fake_attn = self.netg(self.input)
                    self.anomaly, self.anomaly_attn = self.cutout(self.input_2)
                    self.fake_anomaly, self.latent_i_anomaly, self.latent_o_anomaly, self.fake_anomaly_attn = self.netg(self.anomaly)
                else:
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

                time_o = time.time()

                self.an_scores[i*self.batchsize: i*self.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.batchsize: i*self.batchsize + error.size(0)] = self.gt.reshape(error.size(0))

                self.times.append(time_o - time_i)
                
                self.save_test_images = True
                # Save test images.
                if self.save_test_images:
                    dst = os.path.join(self.save_dir, 'test', 'images')
                    dst_ano = os.path.join(self.save_dir, 'test_ano', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    if not os.path.isdir(dst_ano): os.makedirs(dst_ano)
                    real, fake, real_ano, fake_ano = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)
                    
                    vutils.save_image(real_ano, '%s/real_ano_%03d.eps' % (dst_ano, i+1), normalize=True)
                    vutils.save_image(fake_ano, '%s/fake_ano_%03d.eps' % (dst_ano, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            from lib.evaluate import roc
            auc = roc(self.gt_labels, self.an_scores)
            

            ##
            # RETURN
            return auc
    ##
    def plot_loss_histogram(self,loss_list, name):
        from matplotlib import pyplot
        import numpy
        import matplotlib.pyplot as plt
        bins = numpy.linspace(0,18, 100)
        pyplot.hist(loss_list, bins=bins, alpha=0.5, label=name)
        os.makedirs('./runs/detect',exist_ok=True)
        filename = str(name) + '.jpg'
        file_path = os.path.join('./runs/detect',filename)
        plt.savefig(file_path)
        plt.show()
    ##
    #https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib
    def plot_two_loss_histogram(self,normal_list, abnormal_list, name):
        import numpy
        from matplotlib import pyplot
        import matplotlib.pyplot as plt
        bins = numpy.linspace(0, 18, 100)
        pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
        pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
        pyplot.legend(loc='upper right')
        os.makedirs('./runs/detect',exist_ok=True)
        filename = str(name) + '.jpg'
        file_path = os.path.join('./runs/detect',filename)
        plt.savefig(file_path)
        pyplot.show()
        
    ## 
    def Analysis_two_list(self, normal_list, abnormal_list, name, user_loss_list=None):
        import math
        import numpy
        normal_count_list = [0]*len(user_loss_list)
        abnormal_count_list = [0]*len(user_loss_list)
        for i in range(len(normal_list)):
            normal_count_list[int(normal_list[i])]+=1
        print('normal_count_list')
        for i in range(len(normal_count_list)):
            print('{}: {}'.format(i,normal_count_list[i]))
        
        for i in range(len(abnormal_list)):
            abnormal_count_list[int(abnormal_list[i])]+=1
        print('abnormal_count_list')
        for i in range(len(abnormal_count_list)):
            print('{}: {}'.format(i,abnormal_count_list[i]))
        
        overlap_normal_count = 0
        overlap_abnormal_count = 0
        overlap_count = 0
        for i in range(len(normal_count_list)):
            if normal_count_list[i]!=0 and abnormal_count_list[i]!=0:
                overlap_normal_count += normal_count_list[i]
                overlap_abnormal_count += abnormal_count_list[i]
                overlap_count += min(normal_count_list[i],abnormal_count_list[i])
        print('overlap_normal_count: {}'.format(overlap_normal_count))
        print('overlap_abnormal_count: {}'.format(overlap_abnormal_count))
        print('overlap_count: {}'.format(overlap_count))
        
        from matplotlib import pyplot
        bins = numpy.linspace(0, 13, 100)
        pyplot.hist(normal_list, bins, alpha=0.5, label='normal')
        pyplot.hist(abnormal_list, bins, alpha=0.5, label='abnormal')
        pyplot.legend(loc='upper right')
        os.makedirs('./runs/detect',exist_ok=True)
        filename = str(name) + '.jpg'
        file_path = os.path.join('./runs/detect',filename)
        pyplot.savefig(file_path)
        pyplot.show()
        
        if user_loss_list is None:
            normal_acc,abnormal_acc = self.Get_lossTH_Accuracy(normal_count_list,abnormal_count_list)
        else:
            normal_acc,abnormal_acc = self.Get_lossTH_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list, user_loss_list)
        
        return normal_count_list,abnormal_count_list,normal_acc,abnormal_acc
    ##
    def Analysis_two_list_UserDefineLossTH(self, normal_list, abnormal_list, name, user_loss_list=None):
        show_log = False
        import math
        import numpy
        normal_count_list = [0]*len(user_loss_list)
        abnormal_count_list = [0]*len(user_loss_list)
        
        user_loss_list = sorted(user_loss_list)
        
        if show_log:
            print('normal_list : {}'.format(normal_list))
            print('abnormal_list : {}'.format(abnormal_list))
            print('user_loss_list : {}'.format(user_loss_list))
        
        for i in range(len(user_loss_list)):
            for j in range(len(normal_list)):
                if (i+1) < len(user_loss_list):
                    if normal_list[j] >= user_loss_list[i] and  normal_list[j] < user_loss_list[i+1]:
                        normal_count_list[i]+=1
                else:
                    if normal_list[j] >= user_loss_list[i]:
                        normal_count_list[i]+=1
        
        for i in range(len(user_loss_list)):
            for j in range(len(abnormal_list)):
                if (i+1) < len(user_loss_list):
                    if abnormal_list[j] >= user_loss_list[i] and  abnormal_list[j] < user_loss_list[i+1]:
                        abnormal_count_list[i]+=1
                else:
                    if abnormal_list[j] >= user_loss_list[i]:
                        abnormal_count_list[i]+=1
                
        normal_acc,abnormal_acc = self.Get_lossTH_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list, user_loss_list)
        
        print('user_loss_list: {}'.format(user_loss_list))
        
        print('normal_count_list:') 
        for i in range(len(user_loss_list)):
            print('{} : {}'.format(user_loss_list[i], normal_count_list[i]))
            
        print('abnormal_count_list:')
        for i in range(len(user_loss_list)):
            print('{} : {}'.format(user_loss_list[i], abnormal_count_list[i]))
            
            
        #print('normal_count_list: {}'.format(normal_count_list))
        #print('abnormal_count_list: {}'.format(abnormal_count_list))
        
        return normal_count_list,abnormal_count_list,normal_acc,abnormal_acc
    ##
    def Analysis_Accuracy(self, normal_count_list,abnormal_count_list,loss_th=3.0):
        show_log = False
        normal_correct_cnt = 0
        total_normal_cnt = 0
        for i in range(len(normal_count_list)):
            total_normal_cnt+=normal_count_list[i]
            if i < loss_th:
                normal_correct_cnt+=normal_count_list[i]
        if show_log:
            print('normal_correct_cnt: {}'.format(normal_correct_cnt))
            print('total_normal_cnt: {}'.format(total_normal_cnt))
        if total_normal_cnt == 0:
            normal_acc = 0.0
        else:
            normal_acc = float(normal_correct_cnt/total_normal_cnt)
        
        total_abnormal_cnt = 0
        abnormal_correct_cnt = 0
        for i in range(len(abnormal_count_list)):
            total_abnormal_cnt+=abnormal_count_list[i]
            if i >= loss_th:
                abnormal_correct_cnt+=abnormal_count_list[i]
        if show_log:
            print('abnormal_correct_cnt : {}'.format(abnormal_correct_cnt))
            print('total_abnormal_cnt: {}'.format(total_abnormal_cnt))
        if total_abnormal_cnt==0:
            abnormal_acc = 0
        else:
            abnormal_acc = float(abnormal_correct_cnt / total_abnormal_cnt)
        
        
        return normal_acc,abnormal_acc
    
    ##
    def Analysis_Accuracy_UserDefineLossTH(self, normal_count_list,abnormal_count_list,loss_th=3.0, user_loss_list=None):
        show_log = False
        normal_correct_cnt = 0
        total_normal_cnt = 0
        for i in range(len(normal_count_list)):
            total_normal_cnt+=normal_count_list[i]
            if user_loss_list[i] < loss_th:
                normal_correct_cnt+=normal_count_list[i]
        if show_log:
            print('normal_correct_cnt: {}'.format(normal_correct_cnt))
            print('total_normal_cnt: {}'.format(total_normal_cnt))
        if total_normal_cnt == 0:
            normal_acc = 0.0
        else:
            normal_acc = float(normal_correct_cnt/total_normal_cnt)
        
        total_abnormal_cnt = 0
        abnormal_correct_cnt = 0
        for i in range(len(abnormal_count_list)):
            total_abnormal_cnt+=abnormal_count_list[i]
            if user_loss_list[i] >= loss_th:
                abnormal_correct_cnt+=abnormal_count_list[i]
        if show_log:
            print('abnormal_correct_cnt : {}'.format(abnormal_correct_cnt))
            print('total_abnormal_cnt: {}'.format(total_abnormal_cnt))
        if total_abnormal_cnt==0:
            abnormal_acc = 0
        else:
            abnormal_acc = float(abnormal_correct_cnt / total_abnormal_cnt)
        
        
        return normal_acc,abnormal_acc
    
    ##
    def Get_lossTH_Accuracy(self, normal_count_list,abnormal_count_list):
        normal_acc_list,abnormal_acc_list=[0.0]*10,[0.0]*10
        
        for i in range(len(normal_acc_list)):
            normal_acc,abnormal_acc = self.Analysis_Accuracy(normal_count_list,abnormal_count_list,i)
                  
            normal_acc_list[i] = normal_acc
            abnormal_acc_list[i] = abnormal_acc
            
        for i in range(len(normal_acc_list)):
            print('loss {} ,normal acc: {} ,abnormal acc{}'.format(i,normal_acc_list[i],abnormal_acc_list[i]))
            
        return normal_acc,abnormal_acc
    ##
    def Get_lossTH_Accuracy_UserDefineLossTH(self, normal_count_list,abnormal_count_list, user_loss_list):
        normal_acc_list,abnormal_acc_list=[0.0]*len(user_loss_list),[0.0]*len(user_loss_list)
        
        for i in range(len(user_loss_list)):
            normal_acc,abnormal_acc = self.Analysis_Accuracy_UserDefineLossTH(normal_count_list,abnormal_count_list,user_loss_list[i],user_loss_list)
                  
            normal_acc_list[i] = normal_acc
            abnormal_acc_list[i] = abnormal_acc
            
        for i in range(len(user_loss_list)):
            print('loss {} ,normal acc: {} ,abnormal acc{}'.format(user_loss_list[i],normal_acc_list[i],abnormal_acc_list[i]))
            
        return normal_acc,abnormal_acc
    
  