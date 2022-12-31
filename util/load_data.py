# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:31:56 2022

@author: User
"""
import torchvision.transforms as transforms
import torchvision
import torch

def load_data(args,train=True,test=False):
    size = (args.img_size,args.img_size)
    if train:
        img_data = torchvision.datasets.ImageFolder(args.img_dir,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),                                                 
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                    ])
                                                    )
    elif test:
        img_data = torchvision.datasets.ImageFolder(args.img_testdir,
                                                    transform=transforms.Compose([
                                                    transforms.Resize(size),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.Scale(64),
                                                    transforms.CenterCrop(size),                                                 
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                    ])
                                                    )
    if train:
        shuffle=True
        batch_size = args.batch_size
    else:
        shuffle=False
        batch_size = args.test_batchsize
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=batch_size,shuffle=shuffle,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader


def print_parameters(args):
    print('IMAGE_SIZE_H:{}\n IMAGE_SIZE_W:{}\n TRAIN_DATA_DIR:{}\n BATCH_SIZE:{}\n SAVE_MODEL_DIR:{}\n n_epochs:{}\n load weights:{}\n nz:{}\n nc:{}'.format(args.img_size,
                            args.img_size,
                            args.img_dir,
                            args.batch_size,
                            args.save_dir,
                            args.epoch,
                            args.weights,
                            args.nz,
                            args.nc))