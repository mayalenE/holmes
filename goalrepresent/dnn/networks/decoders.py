import math
from abc import ABCMeta

import torch
from torch import nn

import goalrepresent as gr
from goalrepresent.helper.nnmodulehelper import Channelize, conv2d_output_sizes, convtranspose2d_get_output_padding


class BaseDNNDecoder(nn.Module, metaclass=ABCMeta):
    """
    Base Decoder class
    User can specify variable number of convolutional layers to deal with images of different sizes (eg: 3 layers for 32*32 images3 layers for 32*32 images, 6 layers for 256*256 images)
    
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
        default_config.feature_layer = 2
        default_config.hidden_channels = None
        default_config.hidden_dim = None

        return default_config

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        self.output_keys_list = ["z", "gfi", "lfi", "recon_x"]

    def forward(self, z):
        if z.dim() == 2 and type(self).__name__ == "DumoulinDecoder":  # B*n_latents -> B*n_latents*1*1
            z = z.unsqueeze(dim=-1).unsqueeze(dim=-1)

        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(z)

            # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        # decoder output
        decoder_outputs = {"z": z, "gfi": gfi, "lfi": lfi, "recon_x": recon_x}

        return decoder_outputs

    def forward_for_graph_tracing(self, z):
        # global feature map
        gfi = self.efi(z)
        # global feature map
        lfi = self.gfi(gfi)
        # recon_x
        recon_x = self.lfi(lfi)
        return recon_x

    def push_variable_to_device(self, x):
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return x


def get_decoder(model_architecture):
    '''
    model_architecture: string such that the class decoder called is <model_architecture>Decoder
    '''
    return eval("{}Decoder".format(model_architecture))


""" ========================================================================================================================
Decoder Modules 
========================================================================================================================="""


class BurgessDecoder(BaseDNNDecoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNDecoder.__init__(self, config=config, **kwargs)

        # network architecture
        # WARNING: incrementation order follow the encoder top-down order
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
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        output_pads = [None] * self.config.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.config.input_size,
                                                            kernels_size[0],
                                                            strides[0], pads[0])
        for conv_layer_id in range(1, self.config.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[conv_layer_id],
                                                                            feature_map_sizes[conv_layer_id - 1],
                                                                            kernels_size[conv_layer_id],
                                                                            strides[conv_layer_id], pads[conv_layer_id])

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.config.n_latents, hidden_dim), nn.ReLU())
        self.efi.out_connection_type = ("lin", hidden_dim)

        # global feature inverse
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_1_i", nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.gfi.add_module("lin_0_i",
                            nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs),
                                          nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(hidden_channels, h_after_convs, w_after_convs))
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.gfi.out_connection_type = ("conv", hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]), nn.ReLU()))
        self.lfi.add_module("conv_0_i",
                            nn.ConvTranspose2d(hidden_channels, self.config.n_channels, kernels_size[0], strides[0],
                                               pads[0],
                                               output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.config.n_channels)


class HjelmDecoder(BaseDNNDecoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNDecoder.__init__(self, config=config, **kwargs)

        # network architecture
        # WARNING: incrementation order follow the encoder top-down order
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
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        output_pads = [None] * self.config.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.config.input_size,
                                                            kernels_size[0],
                                                            strides[0], pads[0])
        for conv_layer_id in range(1, self.config.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[conv_layer_id],
                                                                            feature_map_sizes[conv_layer_id - 1],
                                                                            kernels_size[conv_layer_id],
                                                                            strides[conv_layer_id], pads[conv_layer_id])

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.config.n_latents, hidden_dim), nn.ReLU())
        self.efi.out_connection_type = ("lin", hidden_dim)

        # global feature inverse
        hidden_channels = int(hidden_channels * math.pow(2, self.config.n_conv_layers - 1))
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_0_i",
                            nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs),
                                          nn.ReLU()))
        self.gfi.add_module("channelize", Channelize(hidden_channels, h_after_convs, w_after_convs))
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU()))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]), nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU()))
            hidden_channels = hidden_channels // 2
        self.lfi.add_module("conv_0_i",
                            nn.ConvTranspose2d(hidden_channels, self.config.n_channels, kernels_size[0], strides[0],
                                               pads[0],
                                               output_padding=output_pads[0]))
        self.lfi.out_connection_type = ("conv", self.config.n_channels)


class DumoulinDecoder(BaseDNNDecoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNDecoder.__init__(self, config=config, **kwargs)

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

        feature_map_sizes = conv2d_output_sizes(self.config.input_size, 2 * self.config.n_conv_layers, kernels_size,
                                                strides, pads,
                                                dils)
        output_pads = [None] * 2 * self.config.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.config.input_size,
                                                            kernels_size[0],
                                                            strides[0], pads[0])
        output_pads[1] = convtranspose2d_get_output_padding(feature_map_sizes[1], feature_map_sizes[0], kernels_size[1],
                                                            strides[1], pads[1])
        for conv_layer_id in range(1, self.config.n_conv_layers):
            output_pads[2 * conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[2 * conv_layer_id],
                                                                                feature_map_sizes[
                                                                                    2 * conv_layer_id - 1],
                                                                                kernels_size[2 * conv_layer_id],
                                                                                strides[2 * conv_layer_id],
                                                                                pads[2 * conv_layer_id])
            output_pads[2 * conv_layer_id + 1] = convtranspose2d_get_output_padding(
                feature_map_sizes[2 * conv_layer_id + 1], feature_map_sizes[2 * conv_layer_id + 1 - 1],
                kernels_size[2 * conv_layer_id + 1], strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1])

        # encoder feature inverse
        hidden_channels = int(hidden_channels * math.pow(2, self.config.n_conv_layers))
        self.efi = nn.Sequential(
            nn.ConvTranspose2d(self.config.n_latents, hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.efi.out_connection_type = ("conv", hidden_channels)

        # global feature inverse
        self.gfi = nn.Sequential()
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2 * conv_layer_id + 1],
                                   strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                   output_padding=output_pads[2 * conv_layer_id + 1]),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[2 * conv_layer_id],
                                   strides[2 * conv_layer_id], pads[2 * conv_layer_id],
                                   output_padding=output_pads[2 * conv_layer_id]),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[2 * conv_layer_id + 1],
                                   strides[2 * conv_layer_id + 1], pads[2 * conv_layer_id + 1],
                                   output_padding=output_pads[2 * conv_layer_id + 1]),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernels_size[2 * conv_layer_id],
                                   strides[2 * conv_layer_id], pads[2 * conv_layer_id],
                                   output_padding=output_pads[2 * conv_layer_id]),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.LeakyReLU(inplace=True),
            ))
            hidden_channels = hidden_channels // 2
        self.lfi.add_module("conv_0_i", nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[1], strides[1], pads[1],
                               output_padding=output_pads[1]),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, self.config.n_channels, kernels_size[0], strides[0], pads[0],
                               output_padding=output_pads[0]),
            nn.Sigmoid()
        ))
        self.lfi.out_connection_type = ("conv", self.config.n_channels)


class MNISTDecoder(BaseDNNDecoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNDecoder.__init__(self, config=config, **kwargs)

        # network architecture
        if ("linear_layers_dim" not in self.config) or (self.config.linear_layers_dim is None):
            self.config.linear_layers_dim = [400, 400]
        self.config.hidden_dim = self.config.linear_layers_dim[-1]
        n_linear_layers = len(self.config.linear_layers_dim)

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.config.n_latents, self.config.hidden_dim), nn.ReLU())
        self.efi.out_connection_type = ("lin", self.config.hidden_dim)

        # global feature inverse
        ## linear layers
        self.gfi = nn.Sequential()
        for linear_layer_id in range(n_linear_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("lin_{}_i".format(linear_layer_id), nn.Sequential(
                nn.Linear(self.config.linear_layers_dim[linear_layer_id],
                          self.config.linear_layers_dim[linear_layer_id - 1]), nn.ReLU()))
        self.gfi.out_connection_type = ("lin", self.config.hidden_dim)

        # local feature inverse
        self.lfi = nn.Sequential()
        for linear_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("lin_{}_i".format(linear_layer_id),
                                nn.Sequential(nn.Linear(self.config.linear_layers_dim[linear_layer_id],
                                                        self.config.linear_layers_dim[linear_layer_id - 1]), nn.ReLU()))
        self.lfi.add_module("lin_0_i",
                            nn.Linear(self.config.hidden_dim,
                                      self.config.n_channels * self.config.input_size[0] * self.config.input_size[1]))
        self.lfi.add_module("channelize",
                            Channelize(self.config.n_channels, self.config.input_size[0], self.config.input_size[1]))
        self.lfi.out_connection_type = ("conv", self.config.n_channels)


class CedricDecoder(BaseDNNDecoder):

    def __init__(self, config=None, **kwargs):
        BaseDNNDecoder.__init__(self, config=config, **kwargs)

        # network architecture
        # WARNING: incrementation order follow the encoder top-down order
        if self.config.hidden_channels is None:
            self.config.hidden_channels = 32
        hidden_channels = self.config.hidden_channels
        if self.config.hidden_dim is None:
            self.config.hidden_dim = 256
        hidden_dim = self.config.hidden_dim
        kernels_size = [5] * self.config.n_conv_layers
        strides = [2] * self.config.n_conv_layers
        pads = [0] * self.config.n_conv_layers
        dils = [1] * self.config.n_conv_layers
        feature_map_sizes = conv2d_output_sizes(self.config.input_size, self.config.n_conv_layers, kernels_size,
                                                strides, pads, dils)
        h_after_convs, w_after_convs = feature_map_sizes[-1]
        output_pads = [None] * self.config.n_conv_layers
        output_pads[0] = convtranspose2d_get_output_padding(feature_map_sizes[0], self.config.input_size,
                                                            kernels_size[0],
                                                            strides[0], pads[0])
        for conv_layer_id in range(1, self.config.n_conv_layers):
            output_pads[conv_layer_id] = convtranspose2d_get_output_padding(feature_map_sizes[conv_layer_id],
                                                                            feature_map_sizes[conv_layer_id - 1],
                                                                            kernels_size[conv_layer_id],
                                                                            strides[conv_layer_id], pads[conv_layer_id])

        hidden_channels = int(hidden_channels * math.pow(2, self.config.n_conv_layers - 1))

        # encoder feature inverse
        self.efi = nn.Sequential(nn.Linear(self.config.n_latents, hidden_dim), nn.PReLU())
        self.efi.out_connection_type = ("lin", hidden_dim)

        # global feature inverse
        ## linear layers
        self.gfi = nn.Sequential()
        self.gfi.add_module("lin_0_i",
                            nn.Sequential(nn.Linear(hidden_dim, hidden_channels * h_after_convs * w_after_convs),
                                          nn.PReLU()))
        self.gfi.add_module("channelize", Channelize(hidden_channels, h_after_convs, w_after_convs))
        ## convolutional layers
        for conv_layer_id in range(self.config.n_conv_layers - 1, self.config.feature_layer + 1 - 1, -1):
            self.gfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]),
                nn.PReLU()))
            hidden_channels = hidden_channels // 2
        self.gfi.out_connection_type = ("conv", hidden_channels)

        # local feature inverse
        self.lfi = nn.Sequential()
        for conv_layer_id in range(self.config.feature_layer + 1 - 1, 0, -1):
            self.lfi.add_module("conv_{}_i".format(conv_layer_id), nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernels_size[conv_layer_id],
                                   strides[conv_layer_id], pads[conv_layer_id],
                                   output_padding=output_pads[conv_layer_id]),
                nn.PReLU()))
            hidden_channels = hidden_channels // 2

        self.lfi.add_module("conv_0_i", nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, self.config.n_channels, kernels_size[0], strides[0], pads[0],
                               output_padding=output_pads[0])))
        self.lfi.out_connection_type = ("conv", self.config.n_channels)
