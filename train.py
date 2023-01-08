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



def main():
    args = get_args()
    train(args)

def train(args): 
    print_parameters(args)
    '''load data'''
    train_loader = load_data(args)
    test_loader = load_data(args,train=False,test=True)
    '''load model'''
    #skip_ganomaly = UGanomaly(args)
    #ganomaly = Ganomaly(args)
    #skip_attention_ganomaly = SAGanomaly(args)
    Skip_SelfAttention_Ganomaly = SSAGanomaly(args)
    #print(Skip_SelfAttention_Ganomaly)
    ''' train epochs'''
    train_epochs(Skip_SelfAttention_Ganomaly,train_loader,test_loader,args)


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
    
    for epoch in range(1, args.epoch+1):
        train_loss = 0.0
        G_loss = 0
        D_loss = 0
        
        
        
        pbar = tqdm(train_loader)
        for images, _  in pbar: 
            images = images.to(device)
            '''inference'''
            outputs = model(images)
            error_g, error_d, fake_img, model_g, model_d = outputs
            loss = error_g + error_d
            
            bar_str = ' epoch:{} loss_g:{:.6f} loss_d:{:.6f}'.format(epoch,error_g.detach().cpu().numpy(),
                                                                     error_d.detach().cpu().numpy())
            PREFIX = color.colorstr(bar_str)
            pbar.desc = f'{PREFIX}'
            ''' sum loss '''
            G_loss += error_g.item()
            D_loss += error_d.item()
            
            avg_g_loss = G_loss/len(train_loader)
            avg_d_loss = D_loss/len(train_loader)
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
        print('auc = {:.6f}'.format(auc))
        
        if auc >= _highest_auc and epoch>1:
            _highest_auc = auc
            torch.save(model_g.state_dict(), SAVE_BEST_MODEL_G_PATH)
            torch.save(model_d.state_dict(), SAVE_BEST_MODEL_D_PATH)
            color_str = 'Save best-model weights complete with auc : {:.6f}'.format(_highest_auc)
            PREFIX = color.colorstr('red', 'bold',color_str)
            print(PREFIX)
            #print('Save best-model weights complete with auc : %.6f'.format(_highest_auc))
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
    parser.add_argument('-imgdir','--img-dir',help='image dir',default=r"C:\factory_data\2022-12-30\crops_line")
    parser.add_argument('-imgtestdir','--img-testdir',help='val dataset',default=r"C:\factory_data\2022-12-30\crops_2cls")
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=32)
    parser.add_argument('-nz','--nz',type=int,help='compress length',default=100)
    parser.add_argument('-nc','--nc',type=int,help='num of channel',default=3)
    parser.add_argument('-lr','--lr',type=float,help='learning rate',default=2e-4)
    parser.add_argument('-batchsize','--batch-size',type=int,help='train batch size',default=64)
    parser.add_argument('-testbatchsize','--test_batchsize',type=int,help='test batch size',default=64)
    parser.add_argument('-savedir','--save-dir',help='save model dir',default=r"C:\GitHub_Code\cuteboyqq\GANomaly\skip-GANOMALY-Pytorch\runs\train\2023-01-08\32-nz100-ngf64-ndf64-Skip-SelfAttention-Ganomaly")
    parser.add_argument('-weights','--weights',help='save model dir',default=r"")
    parser.add_argument('-epoch','--epoch',type=int,help='num of epochs',default=60)
    parser.add_argument('-train','--train',type=bool,help='train model',default=True)
    return parser.parse_args()    

if __name__=="__main__":
    main()
