import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from goalrepresent.helper.nnmodulehelper import Roll, SphericPad

to_PIL = transforms.ToPILImage()
to_Tensor = transforms.ToTensor()

''' ---------------------------------------------
               PREPROCESS DATA HELPER
-------------------------------------------------'''

class RandomGaussianBlur(object):
    def __init__(self, p=0.5, kernel_radius=5, max_sigma=5, n_channels=1):
        self.p = p
        self.kernel_size = 2 * kernel_radius + 1
        self.padding_size = [int((self.kernel_size - 1) / 2)]*4
        self.max_sigma = max_sigma
        self.n_channels = n_channels

    def gaussian_kernel(self, kernel_size, sigma):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, *gaussian_kernel.size())
        gaussian_kernel = gaussian_kernel.repeat(self.n_channels, 1, 1, 1)
        return gaussian_kernel

    def __call__(self, x):
        if np.random.random() < self.p:
            sigma =  int(np.round(np.random.uniform(1.0, self.max_sigma)))
            x = x.view(1, x.size(0), x.size(1), x.size(2))
            x = F.pad(x, pad=self.padding_size, mode='reflect')
            kernel = self.gaussian_kernel(self.kernel_size, sigma)
            x = F.conv2d(x, kernel, groups=self.n_channels).squeeze(0)

        return x



class RandomSphericalRotation(object):
    def __init__(self, p=0.5, max_degrees=20, n_channels=1, img_size=(64, 64)):
        self.p = p
        self.max_degrees = max_degrees
        radius = max(img_size[0], img_size[1]) / 2
        padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
        # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
        self.spheric_pad = SphericPad(
            padding_size=padding_size)
        if n_channels == 1:
            fill = (0,)
        else:
            fill = 0
        self.random_rotation = transforms.RandomRotation(max_degrees, resample=Image.BILINEAR, fill=fill)
        self.center_crop = transforms.CenterCrop(img_size)


    def __call__(self, x):
        if np.random.random() < self.p:
            x = self.spheric_pad(x.view(1, x.size(0), x.size(1), x.size(2))).squeeze(0)
            img_PIL = to_PIL(x)
            img_PIL = self.random_rotation(img_PIL)
            img_PIL = self.center_crop(img_PIL)
            x = to_Tensor(img_PIL)

        return x


class RandomRoll(object):
    def __init__(self, p_x=0.5, p_y=0.5, max_dx=0.5, max_dy=0.5, img_size=(64, 64)):
        self.p_x = p_x
        self.max_dx = max_dx * img_size[0]
        self.roll_x = Roll(shift=0, dim=2)

        self.p_y = p_y
        self.max_dy = max_dy * img_size[1]
        self.roll_y = Roll(shift=0, dim=1)

    def __call__(self, x):
        if np.random.random() < self.p_y:
            shift_y = int(np.round(np.random.uniform(-self.max_dy, self.max_dy)))
            self.roll_y.shift = shift_y
            x = self.roll_y(x)

        if np.random.random() < self.p_x:
            shift_x = int(np.round(np.random.uniform(-self.max_dx, self.max_dx)))
            self.roll_x.shift = shift_x
            x = self.roll_x(x)

        return x


class RandomCenterCrop(object):
    def __init__(self, p=0.5, crop_ratio=(1,5), keep_img_size=True):
        self.p = p
        self.crop_ratio = crop_ratio
        self.keep_img_size = keep_img_size

    def __call__(self, x):
        if np.random.random() < self.p:
            crop_ratio = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        else:
            crop_ratio = 1.0
        return centroid_crop_preprocess(x, crop_ratio, self.keep_img_size)

class Crop_Preprocess(object):
    def __init__(self, crop_type='center', crop_ratio=1):
        if crop_type not in ['center', 'random']:
            raise ValueError('Unknown crop type {!r}'.format(crop_type))
        self.crop_type = crop_type
        self.crop_ratio = crop_ratio

    def __call__(self, x):
        if self.crop_type == 'center':
            return centroid_crop_preprocess(x, self.crop_ratio)
        elif self.crop_type == 'random':
            return random_crop_preprocess(x, self.crop_ratio)


def centroid_crop_preprocess(x, ratio=2, keep_img_size=False):
    '''
    arg: x, tensor 1xHxW 
    '''
    if ratio == 1:
        return x

    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0] / ratio, img_size[1] / ratio)

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
        mY = (img_size[1] - 1) / 2  # the crop is happening in PIL system (so inverse of numpy (x,y))
        mX = (img_size[0] - 1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    # if crop in spherical image
    padding_size = round(max(patch_size[0] / 2, patch_size[1] / 2))
    spheric_pad = SphericPad(padding_size=padding_size)
    mX += padding_size
    mY += padding_size

    j = int(mX - patch_size[0] / 2)
    i = int(mY - patch_size[1] / 2)
    w = patch_size[0]
    h = patch_size[1]
    x = spheric_pad(x.view(1, x.size(0), x.size(1), x.size(2))).squeeze(0)
    x = to_PIL(x)
    if keep_img_size:
        patch = transforms.functional.resized_crop(x, i, j, h, w, img_size)
    else:
        patch = transforms.functional.crop(x, i, j, h, w)
    patch = to_Tensor(patch)

    return patch


def random_crop_preprocess(x, ratio=2):
    '''
    arg: x, tensor 1xHxW 
    '''
    if ratio == 1:
        return x

    img_size = (x.size(1), x.size(2))
    patch_size = (img_size[0] / ratio, img_size[1] / ratio)
    random_crop_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomCrop(patch_size), transforms.ToTensor()])

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
        mY = (img_size[1] - 1) / 2
        mX = (img_size[0] - 1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00
    ## raw set seed
    global_rng_state = random.getstate()
    local_seed = mX * mY
    random.seed(local_seed)

    n_trials = 0
    best_patch_activation = 0
    selected_patch = False

    activation = m00 / (img_size[0] * img_size[1])
    while 1:
        patch = random_crop_transform(x)
        patch_activation = patch.sum(dim=-1).sum(dim=-1) / (patch_size[0] * patch_size[1])

        if patch_activation > (activation * 0.5):
            selected_patch = patch
            break

        if patch_activation >= best_patch_activation:
            best_patch_activation = patch_activation
            selected_patch = patch

        n_trials += 1
        if n_trials == 20:
            break

    ## reput global random state
    random.setstate(global_rng_state)

    return selected_patch
