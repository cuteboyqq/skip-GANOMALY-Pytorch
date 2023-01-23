# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:31:56 2022

@author: User
"""
import torchvision.transforms as transforms
import torchvision
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from util.datasets import get_cifar_anomaly_dataset
from util.datasets import get_mnist_anomaly_dataset

def load_data(args,train=True,test=False,CUSTOM=True):
    size = (args.img_size,args.img_size)
    ## USE CUSTOM Datasets
    if CUSTOM:
        if train:
            ## CUSTOM
            
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
        
    ## USE OPEN SOURCE Datasets (CIFAR10, MNIST)
    else:
        ## CIFAR
        if args.img_dir in ['cifar10']:
            transform = transforms.Compose([transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
            train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
            valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
            train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, train_ds.class_to_idx[args.abnormal_class])
    
        ## MNIST
        elif args.img_dir in ['mnist']:
            transform = transforms.Compose([transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
    
    
            train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
            valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
            train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(args.abnormal_class))
            
        ## DATALOADER
        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
            
    
    if CUSTOM:
        return data_loader
    else:
        return train_dl, valid_dl
        


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