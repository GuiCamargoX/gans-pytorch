import os
import random

import torch
import torchvision as tv
from torch.utils.data import Dataset, TensorDataset

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'mnist': tv.datasets.MNIST
}


def load_dataset(name='mnist', size=(28, 28), val_split=0.25, seed=0):
    """Loads a dataset.
    Args:
        name (str): Name of dataset to be loaded.
        size (tuple): Height and width to be resized.
        val_split (float): Percentage of split for the validation set.
        seed (int): Randomness seed.
    Returns:
        Training, validation and testing sets of loaded dataset.
    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Checks if it is supposed to load custom datasets
    #

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