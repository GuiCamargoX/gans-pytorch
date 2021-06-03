import torchvision as tv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import torch

import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'cifar10': tv.datasets.CIFAR10,
    'mnist': tv.datasets.MNIST
}


def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1
        
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    
    weight = [0] * len(images)
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

def loadAMD(size):
    
    data_transforms = {
        'train': tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets={}
    
    train = tv.datasets.ImageFolder(root=r'./data/amd/train', transform=data_transforms['train'])
    val = tv.datasets.ImageFolder(root=r'./data/amd/test', transform=data_transforms['valid'])   
    
    return train,val, val


def load_dataset(name='mnist', size=(28, 28), val_split=0.25):

    # Loads the training data
    train = DATASETS[name](root='./data/'+name, train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor(),
                                tv.transforms.Resize(size),
                                tv.transforms.Normalize(mean=(0.5), std=(0.5)) ])
                           )

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data/'+name, train=False, download=True,
                          transform=tv.transforms.Compose(
                              [tv.transforms.ToTensor(),
                               tv.transforms.Resize(size),
                               tv.transforms.Normalize(mean=(0.5), std=(0.5)) ])
                          )

    return train, val, test

def load_datasetloader(dataset='mnist', input_size=28, batch=64, num_workers=0, unbalanced=False, seed=0):
    
    # Defining the torch seed
    torch.manual_seed(seed)

    #load dataset
    # Checks if it is supposed to load custom datasets
    if dataset == 'amd':
        train, val, _ = loadAMD(input_size)
    # Check if it is in torchvision 
    elif dataset in DATASETS.keys():
        train, val, _ = load_dataset(name=dataset, size=input_size )
    
    #Pytorch deal with unbalanced data using WeightedRandomSampler
    sampler=None
    if unbalanced:
        weights = make_weights_for_balanced_classes(train.imgs, len(train.classes))                                                                
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

    #Dataloader
    dataloaders_train = DataLoader(train, batch_size=batch , num_workers=num_workers, sampler = sampler )

    dataloaders_valid = DataLoader(val, batch_size=batch , num_workers=num_workers )


    return dataloaders_train, dataloaders_valid, _
