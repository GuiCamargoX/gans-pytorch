import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

#from torchvision.models.inception import inception_v3
from torchvision.models import inception_v3
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import entropy


#Inception Score (IS)
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



#Fréchet Inception Distance (FID)
def FID(gan, dataloader ):

    gen=gan.G
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model = inception_model.eval() # Evaluation mode

    inception_model.fc = torch.nn.Identity()

    fake_features_list = []
    real_features_list = []

    gen.eval()
 
    with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
        #try:
        for real_example, _ in tqdm(dataloader, total=len(dataloader)): # Go by batch
            
            if real_example.shape[1] == 1:
                real_example = real_example.repeat(1,3,1,1)

            real_samples = preprocess( real_example.to(device) )
            real_features = inception_model( real_samples ).detach().to('cpu') # Move features to CPU
            real_features_list.append(real_features)
                #print(2)
            fake_samples =  gan.get_noise(len(real_example) ).to(device)
            fake_samples = gen(fake_samples)
            if fake_samples.shape[1] == 1:
                fake_samples = fake_samples.repeat(1,3,1,1)

            fake_samples = preprocess(fake_samples)
            fake_features = inception_model( fake_samples ).detach().to('cpu')
            fake_features_list.append(fake_features)


    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)

    mu_fake = fake_features_all.mean(0)
    mu_real = real_features_all.mean(0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)

    f_distance = (mu_real - mu_fake).dot(mu_real - mu_fake) + torch.trace(sigma_real) + torch.trace(sigma_fake) - 2*torch.trace(matrix_sqrt(sigma_real @ sigma_fake)) 
    return f_distance.item()

import scipy
# This is the matrix square root function you will be using
def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    '''
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img

import numpy as np
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

#Precision, Recall and F1 Score