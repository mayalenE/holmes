import torch
from torch import nn
from math import floor
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, ToTensor, ToPILImage, RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip, RandomRotation
import numpy as np
import h5py
import math
from numbers import Number
to_tensor = ToTensor()
to_PIL_image = ToPILImage()


from torchvision import transforms
import random

''' ---------------------------------------------
               PYTORCH DATASET HELPERS
-------------------------------------------------'''
class Dataset(Dataset):
    """ Dataset to train auto-encoders representations during exploration"""
    def __init__(self, img_size, preprocess=None, data_augmentation = False):
        
        self.n_images = 0
        self.images = []
        self.labels = []
        
        self.img_size = img_size
        
        self.preprocess = preprocess
        
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            radius = max(self.img_size[0], self.img_size[1]) / 2    
            padding_size = int(np.sqrt(2*np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(padding_size=padding_size) #max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_resized_crop = RandomResizedCrop(size = self.img_size)
            self.random_rotation = RandomRotation(40)
            self.center_crop = CenterCrop(self.img_size)
            self.roll_y = Roll(shift = 0, dim = 1)
            self.roll_x = Roll(shift = 0, dim = 2)
        
    def update(self, n_images, images, labels=None):
        if labels is None:
            labels = torch.Tensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print('ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(n_images, images.shape[0], labels.shape[0]))
        
        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
                # random rolled translation (ie pixels shifted outside appear on the other side of image))
                p_y = p_x = 0.3
                
                if np.random.random() < p_y:
                    ## the maximum translation is of half the image size
                    max_dy = 0.5 * self.img_size[0]
                    shift_y = int(np.round(np.random.uniform(-max_dy, max_dy)))
                    self.roll_y.shift = shift_y
                    img_tensor = self.roll_y(img_tensor)
                
                if np.random.random() < p_x:
                    max_dx = 0.5 * self.img_size[1]
                    shift_x = int(np.round(np.random.uniform(-max_dx, max_dx)))
                    self.roll_y.shift = shift_x
                    img_tensor = self.roll_x(img_tensor)

                # random spherical padding + rotation (avoid "black holes" when rotating)
                p_r = 0.3
                
                if np.random.random() < p_r:
                    img_tensor = self.spheric_pad(img_tensor.view(1, img_tensor.size(0), img_tensor.size(1), img_tensor.size(2))).squeeze(0)
                    img_PIL = to_PIL_image(img_tensor)
                    img_PIL = self.random_rotation(img_PIL)
                    img_PIL = self.center_crop(img_PIL)
                    img_tensor = to_tensor(img_PIL)


                img_PIL = to_PIL_image(img_tensor)
                # random horizontal flip
                img_PIL = self.random_horizontal_flip(img_PIL)
                # random vertical flip
                img_PIL = self.random_vertical_flip(img_PIL)
                # convert back to tensor
                img_tensor = to_tensor(img_PIL)
                
        if self.preprocess:
            img_tensor = self.preprocess(img_tensor)
            
            
         # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1
        
        return {'image':img_tensor, 'label':label}
    
    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images = self.n_images, images = np.stack(self.images), labels = np.asarray(self.labels))
        return


class DatasetHDF5(Dataset):
    """
    Dataset to train auto-encoders representations during exploration from datatsets in hdf5 files.

    TODO: add a cache for loaded objects to be faster (see https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5)
    """

    def __init__(self, filepath, split='train', img_size=None, preprocess=None, data_augmentation=False):

        self.filepath = filepath
        self.split = split

        # HDF5 file isn’t pickleable and to send Dataset to workers’ processes it needs to be serialised with pickle, self.file will be opened in __getitem__ 
        self.data_group = None
        if img_size is not None:
                self.img_size = img_size
        
        with h5py.File(self.filepath, 'r') as file:
            self.n_images = int(file[self.split]['observations'].shape[0])
            self.has_labels = bool('labels' in file[self.split])
            if img_size is None:
                self.img_size = file[self.split]['observations'][0].shape

        self.preprocess = preprocess
        self.data_augmentation = data_augmentation

        if self.data_augmentation:
            radius = max(self.img_size[0], self.img_size[1]) / 2
            padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
            self.spheric_pad = SphericPad(padding_size=padding_size)  # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.random_resized_crop = RandomResizedCrop(size=self.img_size)
            self.random_rotation = RandomRotation(40)
            self.center_crop = CenterCrop(self.img_size)
            self.roll_y = Roll(shift=0, dim=1)
            self.roll_x = Roll(shift=0, dim=2)


    def __len__(self):
        return self.n_images


    def __getitem__(self, idx):
        # open the HDF5 file here and store as the singleton. Do not open it each time as it introduces huge overhead.
        if self.data_group is None:
            self.data_group = h5py.File(self.filepath , "r")[self.split]
        
        # image
        img_tensor = torch.from_numpy(self.data_group['observations'][idx,:,:]).unsqueeze(dim=0).float()    
        
        if self.data_augmentation:
            # random rolled translation (ie pixels shifted outside appear on the other side of image))
            p_y = p_x = 0.3

            if np.random.random() < p_y:
                ## the maximum translation is of half the image size
                max_dy = 0.5 * self.img_size[0]
                shift_y = int(np.round(np.random.uniform(-max_dy, max_dy)))
                self.roll_y.shift = shift_y
                img_tensor = self.roll_y(img_tensor)

            if np.random.random() < p_x:
                max_dx = 0.5 * self.img_size[1]
                shift_x = int(np.round(np.random.uniform(-max_dx, max_dx)))
                self.roll_y.shift = shift_x
                img_tensor = self.roll_x(img_tensor)

            # random spherical padding + rotation (avoid "black holes" when rotating)
            p_r = 0.3

            if np.random.random() < p_r:
                img_tensor = self.spheric_pad(img_tensor.view(1, img_tensor.size(0), img_tensor.size(1), img_tensor.size(2))).squeeze(0)
                img_PIL = to_PIL_image(img_tensor)
                img_PIL = self.random_rotation(img_PIL)
                img_PIL = self.center_crop(img_PIL)
                img_tensor = to_tensor(img_PIL)

            img_PIL = to_PIL_image(img_tensor)
            # random horizontal flip
            img_PIL = self.random_horizontal_flip(img_PIL)
            # random vertical flip
            img_PIL = self.random_vertical_flip(img_PIL)
            # convert back to tensor
            img_tensor = to_tensor(img_PIL)
        
        if self.preprocess:
            img_tensor = self.preprocess(img_tensor) 

        # label
        label = -1
        if self.has_labels:
            tmp_label = self.data_group['labels'][idx]
            if not np.isnan(tmp_label):
                label = int(tmp_label)

        return {'image': img_tensor, 'label': label}


''' ---------------------------------------------
               NN MODULES HELPERS
-------------------------------------------------'''

class Flatten(nn.Module):
    """Flatten the input """
    def forward(self, input):
        return input.view(input.size(0), -1)

'''
class LinearFromFlatten(nn.Module):
    """Flatten the input and then apply a linear module """
    def __init__(self, output_flat_size):
        super(LinearFromFlatten, self).__init__()
        self.output_flat_size = output_flat_size
        
    def forward(self, input):
        input =  input.view(input.size(0), -1) # Batch_size * flatenned_size
        input_flatten_size = input.size(1) 
        Linear = nn.Linear(input_flatten_size, self.output_flat_size)
        return Linear(input)
 '''   
    
class Channelize(nn.Module):
    """Channelize a flatten input to the given (C,H,W) output """
    def __init__(self, n_channels, height, width):
        super(Channelize, self).__init__()
        self.n_channels = n_channels
        self.height = height
        self.width = width
        
    def forward(self, input):
        return input.view(input.size(0), self.n_channels, self.height, self.width)
    
    
class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""
    
    def __init__(self, padding_size):
        super(SphericPad, self).__init__()
        if isinstance(padding_size, int):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = padding_size
        elif isinstance(padding_size, tuple) and len(padding_size) == 2:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
        elif isinstance(padding_size, tuple) and len(padding_size) == 4:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
        else:
            raise ValueError('The padding size shoud be: int, tuple of size 2 or tuple of size 4')

    
    def forward(self, input):
        
        output = torch.cat([input, input[:, :, :self.pad_bottom, :]], dim=2)
        output = torch.cat([output, output[:, :, :, :self.pad_right]], dim=3)
        output = torch.cat([output[:, :, -(self.pad_bottom+self.pad_top):-self.pad_bottom, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_right+self.pad_left):-self.pad_right], output], dim=3)
        
        return output
    
class Roll(nn.Module):
    """Rolls spherically the input with the given padding shit on the given dimension."""
    
    def __init__(self, shift, dim):
        super(Roll, self).__init__()
        self.shift = shift 
        self.dim = dim
        
    def forward(self, input):
        """ Shifts an image by rolling it"""
        if self.shift == 0:
            return input
    
        elif self.shift < 0:
            self.shift = -self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))
            return torch.cat([input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long)), gap], dim = self.dim)
    
        else:
            self.shift = input.size(self.dim) - self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long))
            return torch.cat([gap, input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))], dim = self.dim)
    
          
def conv2d_output_flatten_size(h_w, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the flattened size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print('The number of kernels({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(len(kernels_size), len(strides), len(pads), len(dils), n_conv))
    
    h = h_w[0]
    w = h_w[1]
    
    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = (kernels_size[conv_id], kernels_size[conv_id])
        if type(strides[conv_id]) is not tuple:
            stride = (strides[conv_id], strides[conv_id])
        if type(pads[conv_id]) is not tuple:
            pad = (pads[conv_id], pads[conv_id])
        if type(dils[conv_id]) is not tuple:
            dil = (dils[conv_id], dils[conv_id])
        h = floor( ((h + (2 * pad[0]) - ( dil[0] * (kernel_size[0] - 1) ) - 1 ) / stride[0]) + 1)
        w = floor( ((w + (2 * pad[1]) - ( dil[1] * (kernel_size[1] - 1) ) - 1 ) / stride[1]) + 1)
    return h*w

def conv2d_output_size(h_w, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print('The number of kernels ({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(len(kernels_size), len(strides), len(pads), len(dils), n_conv))
    
    h = h_w[0]
    w = h_w[1]
    
    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = (kernels_size[conv_id], kernels_size[conv_id])
        if type(strides[conv_id]) is not tuple:
            stride = (strides[conv_id], strides[conv_id])
        if type(pads[conv_id]) is not tuple:
            pad = (pads[conv_id], pads[conv_id])
        if type(dils[conv_id]) is not tuple:
            dil = (dils[conv_id], dils[conv_id])
        h = floor( ((h + (2 * pad[0]) - ( dil[0] * (kernel_size[0] - 1) ) - 1 ) / stride[0]) + 1)
        w = floor( ((w + (2 * pad[1]) - ( dil[1] * (kernel_size[1] - 1) ) - 1 ) / stride[1]) + 1)
    return h, w

''' ---------------------------------------------
               PREPROCESS DATA HELPER
-------------------------------------------------'''

def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        
def weights_init_pytorch_(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        m.reset_parameters()
        
def weights_init_xavier_uniform_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_xavier_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_uniform_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def weights_init_custom_uniform_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.uniform_(-1,1)
        m.weight.data.uniform_(-1/(m.weight.size(2)), 1/(m.weight.size(2))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-1/math.sqrt(m.weight.size(0)), 1/math.sqrt(m.weight.size(0))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
            
        
''' ---------------------------------------------
               LOSSES HELPERS
-------------------------------------------------'''

def MSE_loss(recon_x, x, reduction=True): 
    if reduction:
        """ Returns the reconstruction loss (mean squared error) summed on the image dims and averaged on the batch size """
        return F.mse_loss(recon_x, x, size_average=False) / x.size()[0]  
    else:
        return F.mse_loss(recon_x, x, reduce = False)

def BCE_loss(recon_x, x, reduction=True):
    if reduction:
        """ Returns the reconstruction loss (binary cross entropy) summed on the image dims and averaged on the batch size """
        return F.binary_cross_entropy(recon_x, x, size_average=False) / x.size()[0]  
    else:
        return F.binary_cross_entropy(recon_x, x, reduce = False)

def BCE_with_digits_loss(recon_x, x, reduction=True): 
    if reduction:
        """ Returns the reconstruction loss (sigmoid + binary cross entropy) summed on the image dims and averaged on the batch size """
        return F.binary_cross_entropy_with_logits(recon_x, x, size_average=False) / x.size()[0]  
    else:
        return F.binary_cross_entropy_with_logits(recon_x, x, reduce = False)
        

def KLD_loss(mu, logvar, reduction=True):
    if reduction:
        """ Returns the KLD loss D(q,p) where q is N(mu,var) and p is N(0,I) """
        # 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 0 ) / mu.size()[0] #we  average on the batch
        #KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim)
        return KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var
    else:
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss_per_latent_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        #KL-divergence between a diagonal multivariate normal and the standard normal distribution is the sum on each latent dimension
        KLD_loss = torch.sum(KLD_loss_per_latent_dim, dim = 1)
        # we add a regularisation term so that the KLD loss doesnt "trick" the loss by sacrificing one dimension
        KLD_loss_var = torch.var(KLD_loss_per_latent_dim, dim = 1)
        return KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var

def CE_loss(recon_y, y):
    """ Returns the cross entropy loss (softmax + NLLLoss) averaged on the batch size """
    return F.cross_entropy(recon_y, y, size_average=False) / y.size()[0]  

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
        
''' ---------------------------------------------
               PREPROCESS DATA HELPER
-------------------------------------------------'''
class Crop_Preprocess(object):
    def __init__(self, crop_type ='center', crop_ratio = 1):
        if crop_type not in ['center', 'random']:
            raise ValueError('Unknown crop type {!r}'.format(crop_type))
        self.crop_type = crop_type
        self.crop_ratio = crop_ratio
        
    def __call__(self, x): 
        if self.crop_type == 'center':
            return centroid_crop_preprocess(x, self.crop_ratio)
        elif self.crop_type == 'random':
            return centroid_crop_preprocess(x, self.crop_ratio)

def centroid_crop_preprocess(x, ratio = 2):
    '''
    arg: x, tensor 1xHxW 
    '''    
    if ratio==1:
        return x
    
    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0]/ratio, img_size[1]/ratio)  
    
    # crop around center of mass (mY and mX describe the position of the centroid of the image)
    image = x[0].numpy()
    x_grid, y_grid = np.meshgrid(range(img_size[0]), range(img_size[1]))
    y_power1_image = y_grid * image
    x_power1_image = x_grid * image
    ## raw moments
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    if m00 == 0:
        mY = (img_size[1]-1) / 2  # the crop is happening in PIL system (so inverse of numpy (x,y))
        mX = (img_size[0]-1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    # if crop in spherical image
    padding_size = round(max(patch_size[0]/2, patch_size[1]/2))
    spheric_pad = SphericPad(padding_size=padding_size)
    mX += padding_size
    mY += padding_size
    to_PIL = transforms.ToPILImage()
    to_Tensor = transforms.ToTensor()
    
    j = int(mX - patch_size[0]/2) 
    i = int(mY - patch_size[1]/2) 
    w = patch_size[0]
    h = patch_size[1]
    x = spheric_pad(x.view(1, x.size(0), x.size(1), x.size(2))).squeeze(0)
    x = to_PIL(x)
    patch = transforms.functional.crop(x, i, j, h, w)
    patch = to_Tensor(patch)
        
    return patch

def random_crop_preprocess(x, ratio = 2):
    '''
    arg: x, tensor 1xHxW 
    '''    
    if ratio==1:
        return x
    
    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0]/ratio, img_size[1]/ratio) 
    random_crop_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(patch_size),transforms.ToTensor()])
    
    # set the seed as mX*mY for reproducibility (mY and mX describe the position of the centroid of the image)
    image = x[0].numpy()
    x_grid, y_grid = np.meshgrid(range(img_size[0]), range(img_size[1]))
    y_power1_image = y_grid * image
    x_power1_image = x_grid * image
    ## raw moments
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    if m00 == 0:
        mY = (img_size[1]-1) / 2
        mX = (img_size[0]-1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    ## raw set seed
    global_rng_state = random.getstate()
    local_seed = mX*mY
    random.seed(local_seed)

    
    n_trials = 0
    best_patch_activation = 0
    selected_patch = False
    
    activation = m00 / (img_size[0]*img_size[1]) 
    while 1:
        patch = random_crop_transform(x)
        patch_activation = patch.sum(dim=-1).sum(dim=-1) / (patch_size[0]*patch_size[1])
        
        if patch_activation > (activation * 0.5):
            selected_patch = patch
            break
        
        if patch_activation >= best_patch_activation:
            best_patch_activation = patch_activation
            selected_patch = patch
        
        n_trials +=1
        if n_trials == 20:
            break
        
    ## reput global random state
    random.setstate(global_rng_state)
        
    return selected_patch
