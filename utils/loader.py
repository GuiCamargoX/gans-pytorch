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
    'mnist': tv.datasets.MNIST
}


class AmdOdirDataset(Dataset):
    """Fundus Eye dataset."""

    def __init__(self, dataframe, root_dir, split, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = self.preprocess(dataframe)
        
        index_t, index_v = train_test_split(dataframe.index, test_size=0.20, random_state=42)
        if split=='train':
            self.df = self.df.loc[index_t]
        if split=='valid':
            self.df = self.df.loc[index_v]
            #sampling test to have 50% of amd disease
            self.df = self.undersample( self.df )
                
        self.labels = self.df.loc[:,"AMD"].values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        
        image = Image.open(img_name).convert("RGB")
        
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = ( image, label )
        
        return sample
    
    def preprocess(self, df):
        df = df.copy()
        
        df["left_amd"] = df["Left-Diagnostic Keywords"].apply(lambda x: 1 if "age-related" in x.lower() else 0)

        df["right_amd"] = df["Right-Diagnostic Keywords"].apply(lambda x: 1 if "age-related" in x.lower() else 0)
        
        df_left = df[[ "Left-Fundus", "left_amd"] ].rename({"Left-Fundus":'Fundus', "left_amd":'AMD'},axis=1)
        df_right = df[[ "Right-Fundus", "right_amd"] ].rename({"Right-Fundus":'Fundus', "right_amd":'AMD' },axis=1)
        
        return pd.concat([df_left, df_right])
    
    def undersample(self,df):
        # Class count
        count_class_0, count_class_1 = df['AMD'].value_counts()

        # Divide by class
        df_class_0 = df[df['AMD'] == 0]
        df_class_1 = df[df['AMD'] == 1]

        df_class_0_under = df_class_0.sample(count_class_1)
        
        return pd.concat([df_class_0_under, df_class_1], axis=0)



def load_ichallenge(size):
    data_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    amd_dataset = tv.datasets.ImageFolder(root='./data/iChallenge-AMD-Training400/Training400',
                                            transform=data_transform)

    return amd_dataset, None, None


def load_odir5k(size):

    data_transforms = {
        'train': tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.CenterCrop(size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': tv.transforms.Compose([
            tv.transforms.Resize(size),
            tv.transforms.CenterCrop(size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    df = pd.read_excel("./data/ODIR-5k/meta-data.xlsx", index_col='ID')

    image_datasets = {x: AmdOdirDataset(df.copy(), root_dir='./data/ODIR-5K/ODIR-5K_Training_Dataset', split= x,
                                        transform=data_transforms[x] )
                        for x in ['train', 'valid']}

    return image_datasets['train'], image_datasets['valid'], None


def load_dataset(name='mnist', size=(28, 28), val_split=0.25):

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor(),
                                tv.transforms.Resize(size),
                                tv.transforms.Normalize(mean=(0.5), std=(0.5)) ])
                           )

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
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
    if dataset == 'odir5k':
        train, val, _ = load_odir5k(size)
    elif dataset == 'ichallenge':
        train, val, _ = load_ichallenge(size)
    # Check if it is in torchvision 
    elif dataset in DATASETS.keys():
        train, val, _ = load_dataset(name=dataset, size=input_size )
    
    #Pytorch deal with unbalanced data using WeightedRandomSampler
    sampler=None
    if unbalanced:
        label_list = train.labels
        _, counts = np.unique(label_list, return_counts=True)
        weights = 1. / torch.tensor(counts, dtype=torch.float)
        samples_weights = weights[label_list]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    #Dataloader
    dataloaders_train = DataLoader(train, batch_size=batch , num_workers=num_workers )

    dataloaders_valid = DataLoader(val, batch_size=batch , num_workers=num_workers )


    return dataloaders_train, dataloaders_valid, _
