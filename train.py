# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:45:49 2022
@author: User
"""
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
from network import network
import os
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from util import loss
from network.model import Ganomaly
from network.umodel import UGanomaly
from network.SAmodel import SAGanomaly
from network.SSAmodel import SSAGanomaly
from tqdm import tqdm
from util import color
from util.load_data import load_data, print_parameters
from matplotlib import pyplot as plt


def main():
    args = get_args()
    train(args)

def train(args): 
    print_parameters(args)
    #===================
    '''load data'''
    #===================
    if not args.img_dir=='cifar10' and  not args.img_dir=='mnist':
        print("Use custom datasets : {}".format(args.img_dir))
        train_loader = load_data(args)
        test_loader = load_data(args,train=False,test=True)
    else:
        print("Use open source datasets : {}".format(args.img_dir))
        train_loader, test_loader = load_data(args,train=True,test=False,CUSTOM=False)
    #=================== 
    '''load model'''
    #===================
    if args.model == 'ganomaly':
        model_ganomaly_based = Ganomaly(args)
    elif args.model == 'skip-ganomaly':
        model_ganomaly_based = UGanomaly(args)
    else:
        model_ganomaly_based = SAGanomaly(args)
    #ganomaly = Ganomaly(args)
    #skip_attention_ganomaly = SAGanomaly(args)
    #Skip_SelfAttention_Ganomaly = SSAGanomaly(args)
    #print(skip_attention_ganomaly)
    #=====================
    ''' train epochs'''
    #=====================
    train_epochs(model_ganomaly_based,train_loader,test_loader,args)


def train_epochs(model,train_loader,test_loader,args):
    ''' use gpu if available'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _lowest_loss = 600.0
    
    _highest_auc = 0.0
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    SAVE_MODEL_PATH = os.path.join(args.save_dir,"AE_3_best_2.pt")
    SAVE_MODEL_G_PATH = os.path.join(args.save_dir,"netG.pt")
    SAVE_MODEL_D_PATH = os.path.join(args.save_dir,"netD.pt")
    
    
    SAVE_BEST_MODEL_G_PATH = os.path.join(args.save_dir,"best_netG.pt")
    SAVE_BEST_MODEL_D_PATH = os.path.join(args.save_dir,"best_netD.pt")
    
    '''
    ======================
    add log file
    ===================
    '''
    logFileLoc = args.save_dir + os.sep +args.logfile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("model: %s \n nz: %d \n nc: %d \n lr: %f" % (args.model, args.nz, args.nc, args.lr))
        logger.write("\n%s\t\t%s\t\t%s\t\t%s\t\t%s" % ('Epoch', 'DLoss', 'GLoss', 'auc (val)', 'lr'))
    logger.flush()
    
    GlossTr_list=[]
    auc_val_list=[]
    epoches = []
    for epoch in range(1, args.epoch+1):
        train_loss = 0.0
        G_loss = 0
        D_loss = 0
        
        
        
        pbar = tqdm(train_loader)
        for images, _  in pbar: 
            images = images.to(device)
            '''inference'''
            outputs = model(images)
            Skip_SelfAttention_Ganomaly = False
            if Skip_SelfAttention_Ganomaly:
                error_g, error_d, fake_img, model_g, model_d, error_g_attn, error_g_ano_attn = outputs
            else:    
                error_g, error_d, fake_img, model_g, model_d = outputs
            loss = error_g + error_d
            
            
            
            if Skip_SelfAttention_Ganomaly:
                bar_str = ' epoch:{} loss_g:{:.6f} loss_d:{:.6f} loss_g_attn:{:.6f} error_g_ano_attn:{:.6f}'.format(epoch,error_g.detach().cpu().numpy(),
                                                                         error_d.detach().cpu().numpy(),
                                                                         error_g_attn.detach().cpu().numpy(),
                                                                         error_g_ano_attn.detach().cpu().numpy())
            else:
                bar_str = ' epoch:{} loss_g:{:.6f} loss_d:{:.6f} '.format(epoch,error_g.detach().cpu().numpy(),
                                                                         error_d.detach().cpu().numpy())
            PREFIX = color.colorstr(bar_str)
            pbar.desc = f'{PREFIX}'
            ''' sum loss '''
            G_loss += error_g.item()
            D_loss += error_d.item()
            
        avg_g_loss = G_loss/len(train_loader)
        avg_d_loss = D_loss/len(train_loader)
            
        GlossTr_list.append(avg_g_loss)
        epoches.append(epoch)
            
        #train_loss = train_loss/len(train_loader)
        
        color_str = 'Epoch: {} \t Avg_G_loss: {:.6f} Avg_D_loss: {:.6f}'.format(epoch, avg_g_loss,avg_d_loss)
        PREFIX = color.colorstr('green', 'bold',color_str)
        print(PREFIX)
        #print('Epoch: {} \t Avg_G_loss: {:.6f} Avg_D_loss: {:.6f}'.format(epoch, avg_g_loss,avg_d_loss))
        
        torch.save(model_g.state_dict(), SAVE_MODEL_G_PATH)
        torch.save(model_d.state_dict(), SAVE_MODEL_D_PATH)
        print('Save Epoch {} model complete !'.format(epoch))
        #calculate auc
        auc = model.test(test_loader)
        auc_val_list.append(auc)
        print('auc = {:.6f}'.format(auc))
        
        #====write to log.txt====
        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, avg_g_loss, avg_d_loss, auc, args.lr))
        logger.flush()
        
        if auc >= _highest_auc and epoch>1:
            _highest_auc = auc
            torch.save(model_g.state_dict(), SAVE_BEST_MODEL_G_PATH)
            torch.save(model_d.state_dict(), SAVE_BEST_MODEL_D_PATH)
            color_str = 'Save best-model weights complete with auc : {:.6f}'.format(_highest_auc)
            PREFIX = color.colorstr('red', 'bold',color_str)
            print(PREFIX)
            #print('Save best-model weights complete with auc : %.6f'.format(_highest_auc))
        '''
        ====================================
        draw epoch-loss and epoch-auc plots
        Have Errors, so noted now !
        ====================================
        '''
         
        '''
        # Plot the figures per 50 epochs
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(epoches, GlossTr_list)
        ax1.set_title("Average training Gloss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current Gloss")
        print(args.save_dir)
        plt.savefig(args.save_dir,"loss_vs_epochs.png")

        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))

        ax2.plot(epoches, auc_val_list, label="Val auc")
        ax2.set_title("Average AUC vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current AUC")
        plt.legend(loc='lower right')

        plt.savefig(args.save_dir + "auc_vs_epochs.png")

        plt.close('all')
        '''
def compute_loss(outputs,images,criterion):
    gen_imag, latent_i, latent_o = outputs
    loss_con = loss.l2_loss(images, gen_imag)
    loss_enc = loss.l1_loss(latent_i, latent_o)
    loss_sum = loss_enc + 50*loss_con
    return loss_sum


def set_input(input:torch.Tensor):
    """ Set input and ground truth
    Args:
        input (FloatTensor): Input data for batch i.
    """
    input = input.clone()
    with torch.no_grad():
        input.resize_(input[0].size()).copy_(input[0])
       
    return input


def get_args():
    import argparse
    #isize=64, nz=100, nc=3
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-model','--model',help='ganomaly | skip-ganomaly | skip-attention-ganomaly ',default='skip-ganomaly')
    parser.add_argument('-imgdir','--img-dir',help='folder | cifar10 | mnist ',default='cifar10')
    #parser.add_argument('-imgdir','--img-dir',help='folder | cifar10 | mnist ',default=r"C:\factory_data\2022-12-30\crops_line")
    parser.add_argument('--abnormal_class', default='7', help='Normal class idx for mnist and cifar datasets')
    parser.add_argument('-imgtestdir','--img-testdir',help='val dataset',default=r"C:\factory_data\2022-12-30\crops_2cls")
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=32)
    parser.add_argument('-nz','--nz',type=int,help='compress length',default=100)
    parser.add_argument('-nc','--nc',type=int,help='num of channel',default=3)
    parser.add_argument('-lr','--lr',type=float,help='learning rate',default=2e-4)
    parser.add_argument('-batchsize','--batch-size',type=int,help='train batch size',default=64)
    parser.add_argument('-testbatchsize','--test_batchsize',type=int,help='test batch size',default=64)
    parser.add_argument('-savedir','--save-dir',help='save model dir',default=r"./runs/train/2023-04-22/mnist/")
    parser.add_argument('-weights','--weights',help='save model dir',default=r"")
    parser.add_argument('-epoch','--epoch',type=int,help='num of epochs',default=20)
    parser.add_argument('-train','--train',type=bool,help='train model',default=True)
    parser.add_argument('-logfile','--logfile',help='log file',default='log.txt')
    return parser.parse_args()    

if __name__=="__main__":
    main()
