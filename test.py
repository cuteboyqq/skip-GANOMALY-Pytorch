# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:51:45 2022
@author: User
"""
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from network import network
from util import loss
from util import plot
import warnings
from network.model import Ganomaly
from network.umodel import UGanomaly
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-noramldir','--normal-dir',help='image dir',default=r"C:\factory_data\2022-12-30\crops_line")
    parser.add_argument('-abnoramldir','--abnormal-dir',help='image dir',default= r"C:\factory_data\2022-12-30\crops_noline")
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=32)
    parser.add_argument('-nz','--nz',type=int,help='compress size',default=100)
    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-lr','--lr',type=float,help='learning rate',default=2e-4)
    parser.add_argument('-batchsize','--batch-size',type=int,help='train batch size',default=1)
    parser.add_argument('-savedir','--save-dir',help='save model dir',default=r"C:\GitHub_Code\cuteboyqq\GANomaly\skip-GANOMALY-Pytorch\runs\train\2023-01-01\32-nz100-ngf64-ndf64")
    parser.add_argument('-weights','--weights',help='model dir',default= r"C:\GitHub_Code\cuteboyqq\GANomaly\skip-GANOMALY-Pytorch\runs\train\2023-01-01\32-nz100-ngf64-ndf64")
    parser.add_argument('-viewimg','--view-img',action='store_true',help='view images')
    parser.add_argument('-train','--train',action='store_true',help='view images')
    return parser.parse_args()    


def main():
    args = get_args()
    
    test(args)

def test(args):
    args.view_img = False
    if args.view_img:
        BATCH_SIZE_VAL = 1
        SHOW_MAX_NUM = 5
        shuffle = True
    else:
        BATCH_SIZE_VAL = 1
        SHOW_MAX_NUM = 14400
        shuffle = False
    # convert data to torch.FloatTensor
   
    
    test_loader = data_loader(shuffle,args.normal_dir,args)
 
    defeat_loader = data_loader(shuffle,args.abnormal_dir,args)
    # specify loss function
    criterion = nn.MSELoss()
    show_num = 0
    positive_loss, defeat_loss = [],[]
    print('Start test :') 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #model.eval()
    #model = ConvAutoencoder()
    skip_ganomaly = UGanomaly(args)
    #model = network.NetG(isize=IMAGE_SIZE_H, nc=3, nz=100, ngf=64, ndf=64, ngpu=1, extralayers=0)
    #model = torch.load(modelPath).to(device)
    #model.load_state_dict(torch.load(modelPath))
    print('load model weight from {} success'.format(args.weights))
    print('VAL_DATA_DIR : {}'.format(args.normal_dir))
    
    normal_loss = infer(test_loader,SHOW_MAX_NUM,skip_ganomaly,criterion,positive_loss,
            'normal',device,args)
    
    anomaly_loss = infer(defeat_loader,SHOW_MAX_NUM,skip_ganomaly,criterion,defeat_loss,
            'anomaly',device,args)
        
    if not args.view_img:
        loss_list = [0.0,0.5,1.0,1.4,1.6,1.8,1.9,2.0,2.25,2.50,2.75,3.0,4.0,5.0,6.0]
        plot.plot_loss_distribution(SHOW_MAX_NUM,normal_loss,anomaly_loss)
        skip_ganomaly.plot_two_loss_histogram(normal_loss,anomaly_loss,"2023-01-01-skip-ganomaly-histogram")
        skip_ganomaly.Analysis_two_list_UserDefineLossTH(normal_loss,anomaly_loss,"2023-01-01-skip-ganomaly-histogram",loss_list)
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def compute_loss(outputs,images,criterion):
    gen_imag, latent_i, latent_o = outputs
    loss_con = loss.l2_loss(images, gen_imag)
    loss_enc = loss.l1_loss(latent_i, latent_o)
    loss_sum = loss_enc + 50*loss_con
    return loss_sum


def renormalize(tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))

def infer(data_loader,
          SHOW_MAX_NUM,
          model,
          criterion,
          loss_list,
          data_type,
          device,
          args
          ):
    show_num = 0
    model.eval()
    #with torch.no_grad():
    
    dataiter = iter(data_loader)
    cnt=1
    
    while(show_num < SHOW_MAX_NUM):
        images, labels = dataiter.next()
        print('{} Start {} AE:'.format(show_num,data_type))
        # get sample outputs
        images = images.to(device)
        outputs = model(images)
        #gen_imag, latent_i, latent_o = outputs
        error_g, error_d, fake_img, model_g, model_d = outputs
        loss = error_g
        #loss = compute_loss(outputs,images,criterion)
        loss_list.append(loss.cpu().detach().numpy())
        print('loss : {}'.format(loss))
        
        #unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #images = unorm(images)
        #fake_img = unorm(fake_img.data)
        images = renormalize(images)
        fake_img = renormalize(fake_img)
        
        
        
        #images = images.view(args.batch_size, 3, args.img_size, args.img_size)
        images = images.cpu().numpy()
        fake_img = fake_img.view(args.batch_size, 3, args.img_size, args.img_size)
        fake_img = fake_img.cpu().detach().numpy()
        
       
        if args.view_img:
            import os
            os.makedirs('./runs/detect',exist_ok=True)
            plts = plot.plot_images(images,fake_img)
            if data_type=='normal':
                file_name = 'infer_normal_loss_' + str(loss.detach().cpu().numpy()) + "_" +  str(cnt) + '.jpg'
            else:
                file_name = 'infer_abnormal_loss_' + str(loss.detach().cpu().numpy()) + "_" + str(cnt) + '.jpg'
            file_path = os.path.join('./runs/detect',file_name)
            plts.savefig(file_path)
            plts.show()
            cnt+=1
        show_num+=1
    return loss_list

def data_loader(shuffle,VAL_DATA_DIR,args):
    size = (args.img_size,args.img_size)
    img_test_data = torchvision.datasets.ImageFolder(VAL_DATA_DIR,
                                                transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),
                                
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #GANomaly parameter
                                                    ])
                                                )

    print('img_test_data length : {}'.format(len(img_test_data)))

    test_loader = torch.utils.data.DataLoader(img_test_data, batch_size=args.batch_size,shuffle=shuffle,drop_last=True)
    print('test_loader length : {}'.format(len(test_loader)))
    
    return test_loader


        
if __name__=="__main__":
    
    main()