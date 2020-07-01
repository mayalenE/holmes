import importlib

from torch import nn

from .. import helper

EPS = 1e-12

        
""" ========================================================================================================================
Decoder Modules 
========================================================================================================================="""
# All decoders should be called Encoder<model_architecture>
def get_decoder(model_architecture):
    model_architecture = model_architecture.lower().capitalize()
    module_name = 'autodisc.representations.static.pytorchnnrepresentation.models.decoders'
    module = importlib.import_module(module_name)
    class_name = "Decoder{}".format(model_architecture)
    decoder = getattr(module, class_name)
    return decoder


class DecoderBurgess (nn.Module):
    """ 
    Extended Decoder of the model proposed in Burgess, Christopher P., et al. "Understanding disentangling in $\beta$-VAE." 
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32 images3 layers for 32*32 images, 6 layers for 256*256 images)
    
    Parameters
    ----------
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    """
        
    def __init__(self,  n_channels = 1, input_size = (64,64), n_conv_layers = 4, n_latents = 10, **kwargs):
        super(DecoderBurgess, self).__init__()
        
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
        
        self.decoder = nn.Sequential()
        
        # linear layers
        self.decoder.add_module("lin_1", nn.Sequential(nn.Linear(self.n_latents, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_2", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.decoder.add_module("lin_3", nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs), nn.ReLU()))
        self.decoder.add_module("channelize", helper.Channelize(hidden_channels,  h_after_convs, w_after_convs))
                
        # convolution layers
        for conv_layer_id in range(0, self.n_conv_layers-1):
            # For convTranspose2d the padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
            self.decoder.add_module("convT_{}".format(conv_layer_id+1), nn.Sequential(nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id], pads[conv_layer_id]), nn.ReLU()))
        self.decoder.add_module("convT_{}".format(self.n_conv_layers), nn.ConvTranspose2d(hidden_channels, self.n_channels, kernels_size[self.n_conv_layers-1], strides[self.n_conv_layers-1], pads[self.n_conv_layers-1]))

        
    def forward(self, z):
        return self.decoder(z)