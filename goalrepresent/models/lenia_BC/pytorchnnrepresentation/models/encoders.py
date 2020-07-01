import importlib

import torch
from torch import nn

from .. import helper

EPS = 1e-12

        
""" ========================================================================================================================
Encoder Modules 
========================================================================================================================="""
# All encoders should be called Encoder<model_architecture>
def get_encoder(model_architecture):
    model_architecture = model_architecture.lower().capitalize()
    module_name = 'autodisc.representations.static.pytorchnnrepresentation.models.encoders'
    module = importlib.import_module(module_name)
    class_name = "Encoder{}".format(model_architecture)
    encoder = getattr(module, class_name)
    return encoder


class EncoderBurgess (nn.Module):
    """ 
    Extended Encoder of the model proposed in Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE."
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32, 6 layers for 256*256 images)
    
    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
    - 2 fully connected layers (each of 256 units)
    - Latent distribution:
        - 1 fully connected layer of 2*n_latents units (log variance and mean for Gaussians distributions)
    """
        
    def __init__(self, n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10,  **kwargs):
        super(EncoderBurgess, self).__init__()
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents

        # network architecture
        hidden_channels = 32
        hidden_dim = 256
        kernels_size=[4]*self.n_conv_layers
        strides=[2]*self.n_conv_layers
        pads=[1]*self.n_conv_layers
        dils=[1]*self.n_conv_layers
        h_after_convs, w_after_convs = helper.conv2d_output_size(self.input_size, self.n_conv_layers, kernels_size, strides, pads, dils)
        
        self.encoder = nn.Sequential()
        
        # convolution layers
        self.encoder.add_module("conv_{}".format(1), nn.Sequential(nn.Conv2d(self.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]), nn.ReLU()))
        for conv_layer_id in range(1, self.n_conv_layers):
            self.encoder.add_module("conv_{}".format(conv_layer_id+1), nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.encoder.add_module("flatten", helper.Flatten())
        
        # linear layers
        self.encoder.add_module("lin_1", nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim), nn.ReLU()))
        self.encoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.encoder.add_module("mu_logvar_gen", nn.Linear(hidden_dim, 2 * self.n_latents))

        
    def forward(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)
