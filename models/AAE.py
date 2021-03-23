import argparse
import os
import numpy as np
import math
import itertools
from torch import Tensor

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tools import utils


def reparameterization(mu, logvar, z):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), z)))).cuda()
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self, input_dim=100, input_size=32):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, self.input_dim )
        self.logvar = nn.Linear(512, self.input_dim )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z=self.input_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, input_dim=1, input_size=32):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784 ),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim=1, input_size=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class AAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62

        # load dataset
        self.data_loader =  args.dataloader
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        #self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        #self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        #self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        #self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))        

        # Initialize generator and discriminator
        self.encoder = Encoder(input_dim=self.z_dim, input_size=self.input_size)
        self.decoder = Decoder(input_dim=self.z_dim, input_size=self.input_size)
        self.discriminator = Discriminator(input_dim=self.z_dim, input_size=self.input_size)

        # Optimizers
        self.G_optimizer = optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2)
        )
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2) )


        if self.gpu_mode:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarial_loss = nn.BCELoss().cuda()
            self.pixelwise_loss = nn.L1Loss().cuda()
        else:
            self.adversarial_loss = nn.BCELoss()
            self.pixelwise_loss = nn.L1Loss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.encoder)
        utils.print_network(self.discriminator)
        print('-----------------------------------------------')

    
    def visualize_results(self, n_row, batches_done):
        """Saves a grid of generated digits"""
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim)))).cuda()
        gen_imgs = self.decoder(z)
        save_image(gen_imgs.data, "data/%d.png" % batches_done, nrow=n_row, normalize=True)

    def train(self):
        # ----------
        #  Training
        # ----------
        # Use binary cross-entropy loss

        for epoch in range(self.epoch):
            for i, (imgs, _) in enumerate(self.data_loader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

                # Configure input
                real_imgs = Variable(imgs).cuda()

                # -----------------
                #  Train Generator
                # -----------------

                self.G_optimizer.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                print( decoded_imgs.shape )
                print( real_imgs.shape )
                g_loss = 0.001 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + 0.999 * self.pixelwise_loss( decoded_imgs, real_imgs )

                g_loss.backward()
                self.G_optimizer.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.D_optimizer.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.z_dim)))).cuda()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.D_optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epoch, i, len(self.data_loader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.data_loader) + i
                if batches_done % 400 == 0:
                    self.visualize_results(n_row=10, batches_done=batches_done)
