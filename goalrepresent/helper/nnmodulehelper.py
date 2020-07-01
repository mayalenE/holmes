from math import floor

import torch
from torch import nn

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
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.n_channels, self.height, self.width)


class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size):
        nn.Module.__init__(self)
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
        output = torch.cat([output[:, :, -(self.pad_bottom + self.pad_top):-self.pad_bottom, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_right + self.pad_left):-self.pad_right], output], dim=3)

        return output


class Roll(nn.Module):
    """Rolls spherically the input with the given padding shit on the given dimension."""

    def __init__(self, shift, dim):
        nn.Module.__init__(self)
        self.shift = shift
        self.dim = dim

    def forward(self, input):
        """ Shifts an image by rolling it"""
        if self.shift == 0:
            return input

        elif self.shift < 0:
            self.shift = -self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))
            return torch.cat(
                [input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long)), gap],
                dim=self.dim)

        else:
            self.shift = input.size(self.dim) - self.shift
            gap = input.index_select(self.dim, torch.arange(self.shift, input.size(self.dim), dtype=torch.long))
            return torch.cat([gap, input.index_select(self.dim, torch.arange(self.shift, dtype=torch.long))],
                             dim=self.dim)


def conv2d_output_sizes(h_w, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print(
        'The number of kernels ({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(
            len(kernels_size), len(strides), len(pads), len(dils), n_conv))

    h = h_w[0]
    w = h_w[1]
    output_sizes = []

    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = (kernels_size[conv_id], kernels_size[conv_id])
        if type(strides[conv_id]) is not tuple:
            stride = (strides[conv_id], strides[conv_id])
        if type(pads[conv_id]) is not tuple:
            pad = (pads[conv_id], pads[conv_id])
        if type(dils[conv_id]) is not tuple:
            dil = (dils[conv_id], dils[conv_id])
        h = floor(((h + (2 * pad[0]) - (dil[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
        w = floor(((w + (2 * pad[1]) - (dil[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
        output_sizes.append((h, w))
    return output_sizes


def convtranspose2d_get_output_padding(h_w_in, h_w_out, kernel_size=1, stride=1, pad=0):
    h_in = h_w_in[0]
    w_in = h_w_in[1]
    h_out = h_w_out[0]
    w_out = h_w_out[1]

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)

    out_p_h = h_out + 2 * pad[0] - kernel_size[0] - (h_in - 1) * stride[0]
    out_p_w = w_out + 2 * pad[1] - kernel_size[1] - (w_in - 1) * stride[1]

    return (out_p_h, out_p_w)
