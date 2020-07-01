import math
from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import nn

import goalrepresent as gr
from goalrepresent.helper.nnmodulehelper import Flatten, conv2d_output_sizes


class BaseDNNEncoder(nn.Module, gr.BaseEncoder, metaclass=ABCMeta):
    """
    Base Encoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32, 6 layers for 256*256 images)
    
    Parameters
    ----------
    n_channels: number of channels
    input_size: size of input image
    n_conv_layers: desired number of convolutional layers
    n_latents: dimensionality of the infered latent output.
    """

    @staticmethod
    def default_config():
        default_config = gr.Config()

        default_config.n_channels = 1
        default_config.input_size = (64, 64)
        default_config.n_conv_layers = 4
        default_config.n_latents = 10
        default_config.encoder_conditional_type = "gaussian"
        default_config.feature_layer = 2
        default_config.hidden_channels = None
        default_config.hidden_dim = None

        # add attention layer
        default_config.use_attention = False

        return default_config

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        self.output_keys_list = ["x", "lf", "gf", "z"]
        if self.config.encoder_conditional_type == "gaussian":
            self.output_keys_list += ["mu", "logvar"]

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        # local feature map
        lf = self.lf(x)
        # global feature map
        gf = self.gf(lf)

        encoder_outputs = {"x": x, "lf": lf, "gf": gf}

        # encoding
        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
            if z.ndim > 2:
                mu = mu.squeeze(dim=-1).squeeze(dim=-1)
                logvar = logvar.squeeze(dim=-1).squeeze(dim=-1)
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs.update({"z": z, "mu": mu, "logvar": logvar})
        elif self.config.encoder_conditional_type == "deterministic":
            z = self.ef(gf)
            if z.ndim > 2:
                z = z.squeeze(dim=-1).squeeze(dim=-1)
            encoder_outputs.update({"z": z})

        # attention features
        if self.config.use_attention:
            af = self.af(gf)
            af = F.normalize(af, p=2)
            encoder_outputs.update({"af": af})

        return encoder_outputs

    def forward_for_graph_tracing(self, x):
        lf = self.lf(x)
        gf = self.gf(lf)
        if self.config.encoder_conditional_type == "gaussian":
            mu, logvar = torch.chunk(self.ef(gf), 2, dim=1)
            z = self.reparameterize(mu, logvar)
        else:
            z = self.ef(gf)
        return z, lf

    def push_variable_to_device(self, x):
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calc_embedding(self, x):
        encoder_outputs = self.forward(x)
        return encoder_outputs["z"]


def get_encoder(model_architecture):
    """
    model_architecture: string such that the class encoder called is <model_architecture>Encoder
    """
    return eval("{}Encoder".format(model_architecture))


""" ========================================================================================================================
Encoder Modules 
========================================================================================================================="""


class BurgessEncoder(BaseDNNEncoder):
    """ 
    Extended Encoder of the model proposed in Burgess et al. "Understanding disentangling in $\beta$-VAE"
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
    - 2 fully connected layers (each of 256 units)
    - Latent distribution:
        - 1 fully connected layer of 2*n_latents units (log variance and mean for Gaussians distributions)
    """

    def __init__(self, config=None, **kwargs):
        BaseDNNEncoder.__init__(self, config=config, **kwargs)

        # need square image input
        assert self.config.input_size[0] == self.config.input_size[1], "BurgessEncoder needs a square image input size"

        # network architecture
        if self.config.hidden_channels is None:
            self.config.hidden_channels = 32
        hidden_channels = self.config.hidden_channels
        if self.config.hidden_dim is None:
            self.config.hidden_dim = 256
        hidden_dim = self.config.hidden_dim
        kernels_size = [4] * self.config.n_conv_layers
        strides = [2] * self.config.n_conv_layers
        pads = [1] * self.config.n_conv_layers
        dils = [1] * self.config.n_conv_layers

        # feature map size
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)

        # local feature
        self.local_feature_shape = (
            hidden_channels, feature_map_sizes[self.config.feature_layer][0],
            feature_map_sizes[self.config.feature_layer][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(0), nn.Sequential(
                    nn.Conv2d(self.config.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]),
                    nn.ReLU()))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id],
                              pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.lf.out_connection_type = ("conv", hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.feature_layer + 1, self.config.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id], strides[conv_layer_id],
                          pads[conv_layer_id], dils[conv_layer_id]), nn.ReLU()))
        self.gf.add_module("flatten", Flatten())
        ## linear layers
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.gf.add_module("lin_0",
                           nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim),
                                         nn.ReLU()))
        self.gf.add_module("lin_1", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.gf.out_connection_type = ("lin", hidden_dim)

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(hidden_dim, 2 * self.config.n_latents))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(hidden_dim, self.config.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.config.use_attention:
            self.add_module("af", nn.Linear(hidden_dim, 4 * self.config.n_latents))


class HjelmEncoder(BaseDNNEncoder):
    """ 
    Extended Encoder of the model proposed in Hjelm et al. "Learning deep representations by mutual information estimation and maximization"
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional layers (64-138-256-512 channels for 64*64 image), (4 x 4 kernel), (stride of 2) => for a MxM feature map
    - 1 fully connected layers (1024 units)
    - Latent distribution:
        - 1 fully connected layer of n_latents units
    """

    def __init__(self, config=None, **kwargs):
        BaseDNNEncoder.__init__(self, config=config, **kwargs)

        # need square image input
        assert self.config.input_size[0] == self.config.input_size[1], "HjlemEncoder needs a square image input size"

        # network architecture
        if self.config.hidden_channels is None:
            self.config.hidden_channels = int(math.pow(2, 9 - int(math.log(self.config.input_size[0], 2)) + 3))
        hidden_channels = self.config.hidden_channels
        if self.config.hidden_dim is None:
            self.config.hidden_dim = 1024
        hidden_dim = self.config.hidden_dim
        kernels_size = [4] * self.config.n_conv_layers
        strides = [2] * self.config.n_conv_layers
        pads = [1] * self.config.n_conv_layers
        dils = [1] * self.config.n_conv_layers

        # feature map size
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)

        # local feature 
        ## convolutional layers
        self.local_feature_shape = (
            int(hidden_channels * math.pow(2, self.config.feature_layer)),
            feature_map_sizes[self.config.feature_layer][0],
            feature_map_sizes[self.config.feature_layer][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(0), nn.Sequential(
                    nn.Conv2d(self.config.n_channels, hidden_channels, kernels_size[0], strides[0], pads[0], dils[0]),
                    nn.ReLU()))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels * 2, kernels_size[conv_layer_id], strides[conv_layer_id],
                              pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels * 2),
                    nn.ReLU()))
                hidden_channels *= 2
        self.lf.out_connection_type = ("conv", hidden_channels)

        # global feature 
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.feature_layer + 1, self.config.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 2, kernels_size[conv_layer_id], strides[conv_layer_id],
                          pads[conv_layer_id], dils[conv_layer_id]), nn.BatchNorm2d(hidden_channels * 2), nn.ReLU()))
            hidden_channels *= 2
        self.gf.add_module("flatten", Flatten())
        ## linear layers
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.gf.add_module("lin_0",
                           nn.Sequential(nn.Linear(hidden_channels * h_after_convs * w_after_convs, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim), nn.ReLU()))
        self.gf.out_connection_type = ("lin", hidden_dim)

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(hidden_dim, 2 * self.config.n_latents))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(hidden_dim, self.config.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.config.use_attention:
            self.add_module("af", nn.Linear(hidden_dim, 4 * self.config.n_latents))

    def forward(self, x):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and x.size(0) == 1:
            self.eval()
            encoder_outputs = BaseDNNEncoder.forward(self, x)
            self.train()
        else:
            encoder_outputs = BaseDNNEncoder.forward(self, x)
        return encoder_outputs


class DumoulinEncoder(BaseDNNEncoder):
    """ 
    Some Alexnet-inspired encoder with BatchNorm and LeakyReLU as proposed in Dumoulin et al. "Adversarially learned inference"
    
    Model Architecture (transposed for decoder)
    ------------
    - Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(conv_layer_id) channels), (3 x 3 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (2*2^(conv_layer_id+1) channels), (4 x 4 kernel), (stride of 2), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
    - 1 Convolutional blocks composed of:
        - 1 convolutional layer (2*2^(n_conv_layers) channels), (1 x 1 kernel), (stride of 1), (padding of 1)
        - 1 BatchNorm layer
        - 1 LeakyReLU layer
        - 1 convolutional layer (n_latents channels), (1 x 1 kernel), (stride of 1), (padding of 1)
    """

    def __init__(self, config=None, **kwargs):
        BaseDNNEncoder.__init__(self, config=config, **kwargs)

        # need square and power of 2 image size input
        power = math.log(self.config.input_size[0], 2)
        assert (power % 1 == 0.0) and (power > 3), "Dumoulin Encoder needs a power of 2 as image input size (>=16)"
        assert self.config.input_size[0] == self.config.input_size[
            1], "Dumoulin Encoder needs a square image input size"

        assert self.config.n_conv_layers == power - 2, "The number of convolutional layers in DumoulinEncoder must be log(input_size, 2) - 2 "

        # network architecture
        if self.config.hidden_channels is None:
            self.config.hidden_channels = int(512 // math.pow(2, self.config.n_conv_layers))
        hidden_channels = self.config.hidden_channels
        kernels_size = [4, 4] * self.config.n_conv_layers
        strides = [1, 2] * self.config.n_conv_layers
        pads = [0, 1] * self.config.n_conv_layers
        dils = [1, 1] * self.config.n_conv_layers

        # feature map size
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, 2 * self.config.n_conv_layers, kernels_size,
                                                strides, pads,
                                                dils)

        # local feature 
        ## convolutional layers
        self.local_feature_shape = (
            int(hidden_channels * math.pow(2, self.config.feature_layer + 1)),
            feature_map_sizes[2 * self.config.feature_layer + 1][0],
            feature_map_sizes[2 * self.config.feature_layer + 1][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    nn.Conv2d(self.config.n_channels, hidden_channels, kernels_size[2 * conv_layer_id],
                              strides[2 * conv_layer_id], pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(hidden_channels, 2 * hidden_channels, kernels_size[2 * conv_layer_id + 1],
                              strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1], dils[2 * conv_layer_id + 1]),
                    nn.BatchNorm2d(2 * hidden_channels),
                    nn.LeakyReLU(inplace=True)
                ))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2 * conv_layer_id],
                              strides[2 * conv_layer_id], pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(hidden_channels, 2 * hidden_channels, kernels_size[2 * conv_layer_id + 1],
                              strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1], dils[2 * conv_layer_id + 1]),
                    nn.BatchNorm2d(2 * hidden_channels),
                    nn.LeakyReLU(inplace=True)
                ))
            hidden_channels *= 2
        self.lf.out_connection_type = ("conv", hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.feature_layer + 1, self.config.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernels_size[2 * conv_layer_id], strides[2 * conv_layer_id],
                          pads[2 * conv_layer_id], dils[2 * conv_layer_id]),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_channels, 2 * hidden_channels, kernels_size[2 * conv_layer_id + 1],
                          strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1], dils[2 * conv_layer_id + 1]),
                nn.BatchNorm2d(2 * hidden_channels),
                nn.LeakyReLU(inplace=True)
            ))
            hidden_channels *= 2
        self.gf.out_connection_type = ("conv", hidden_channels)

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_channels, 2 * self.config.n_latents, kernel_size=1, stride=1)
            ))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.config.n_latents, kernel_size=1, stride=1)
            ))

        # attention feature
        if self.config.use_attention:
            self.add_module("af", nn.Linear(hidden_dim, 4 * self.config.n_latents))

    def forward(self, x):
        # batch norm cannot deal with batch_size 1 in train mode
        if self.training and x.size(0) == 1:
            self.eval()
            encoder_outputs = BaseDNNEncoder.forward(x)
            self.train()
        else:
            encoder_outputs = BaseDNNEncoder.forward(x)
        return encoder_outputs


class MNISTEncoder(BaseDNNEncoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNEncoder.__init__(self, config=config, **kwargs)

        # need square image input
        assert self.config.input_size[0] == self.config.input_size[1], "MNISTEncoder needs a square image input size"

        # network architecture
        if ("linear_layers_dim" not in self.config) or (self.config.linear_layers_dim is None):
            self.config.linear_layers_dim = [400, 400]
        self.config.hidden_dim = self.config.linear_layers_dim[-1]
        n_linear_layers = len(self.config.linear_layers_dim)

        # local feature
        self.local_feature_shape = self.config.linear_layers_dim[self.config.feature_layer]

        self.lf = nn.Sequential()
        self.lf.add_module("flatten", Flatten())
        for linear_layer_id in range(self.config.feature_layer + 1):
            if linear_layer_id == 0:
                self.lf.add_module("lin_{}".format(linear_layer_id), nn.Sequential(
                    nn.Linear(self.config.n_channels * self.config.input_size[0] * self.config.input_size[1],
                              self.config.linear_layers_dim[linear_layer_id]), nn.ReLU()))
            else:
                self.lf.add_module("lin_{}".format(linear_layer_id), nn.Sequential(
                    nn.Linear(self.config.linear_layers_dim[linear_layer_id - 1],
                              self.config.linear_layers_dim[linear_layer_id]), nn.ReLU()))
        self.lf.out_connection_type = ("lin", self.local_feature_shape)

        # global feature
        self.gf = nn.Sequential()
        for linear_layer_id in range(self.config.feature_layer + 1, n_linear_layers):
            self.gf.add_module("lin_{}".format(linear_layer_id), nn.Sequential(
                nn.Linear(self.config.linear_layers_dim[linear_layer_id - 1],
                          self.config.linear_layers_dim[linear_layer_id]), nn.ReLU()))
        self.gf.out_connection_type = ("lin", self.config.hidden_dim)

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(self.config.hidden_dim, 2 * self.config.n_latents))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(self.config.hidden_dim, self.config.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.config.use_attention:
            self.add_module("af", nn.Linear(hidden_dim, 4 * self.config.n_latents))


class CedricEncoder(BaseDNNEncoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNEncoder.__init__(self, config=config, **kwargs)

        # need square image input
        assert self.config.input_size[0] == self.config.input_size[1], "CedricEncoder needs a square image input size"

        # network architecture
        if self.config.hidden_channels is None:
            self.config.hidden_channels = 32
        hidden_channels = self.config.hidden_channels
        if self.config.hidden_dim is None:
            self.config.hidden_dim = 64
        hidden_dim = self.config.hidden_dim
        kernels_size = [5] * self.config.n_conv_layers
        strides = [2] * self.config.n_conv_layers
        pads = [0] * self.config.n_conv_layers
        dils = [1] * self.config.n_conv_layers

        # feature map size
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)

        # local feature
        self.local_feature_shape = (
            hidden_channels, feature_map_sizes[self.config.feature_layer][0],
            feature_map_sizes[self.config.feature_layer][1])
        self.lf = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1):
            if conv_layer_id == 0:
                self.lf.add_module("conv_{}".format(0), nn.Sequential(
                    nn.Conv2d(self.config.n_channels, hidden_channels, kernels_size[0]),
                    nn.PReLU(),
                    nn.MaxPool2d(2, stride=strides[0])))
            else:
                self.lf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                    nn.Conv2d(hidden_channels // 2, hidden_channels, kernels_size[conv_layer_id]),
                    nn.PReLU(),
                    nn.MaxPool2d(2, stride=strides[conv_layer_id])))
            hidden_channels *= 2
        self.lf.out_connection_type = ("conv", hidden_channels)

        # global feature
        self.gf = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.feature_layer + 1, self.config.n_conv_layers):
            self.gf.add_module("conv_{}".format(conv_layer_id), nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels, kernels_size[conv_layer_id]),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=strides[conv_layer_id])))
            hidden_channels *= 2
        self.gf.add_module("flatten", Flatten())
        ## linear layers
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        self.gf.add_module("lin_0",
                           nn.Sequential(nn.Linear(hidden_channels // 2 * h_after_convs * w_after_convs, hidden_dim),
                                         nn.PReLU()))

        # encoding feature
        if self.config.encoder_conditional_type == "gaussian":
            self.add_module("ef", nn.Linear(hidden_dim, 2 * self.config.n_latents))
        elif self.config.encoder_conditional_type == "deterministic":
            self.add_module("ef", nn.Linear(hidden_dim, self.config.n_latents))
        else:
            raise ValueError("The conditional type must be either gaussian or deterministic")

        # attention feature
        if self.config.use_attention:
            self.add_module("af", nn.Linear(hidden_dim, 4 * self.config.n_latents))
