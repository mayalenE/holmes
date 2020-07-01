import os
from copy import deepcopy

import numpy as np
import torch

import goalrepresent as gr
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel


class BCSpectrumFourierModel():

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.set_BC_range = True
        return default_config

    def __init__(self, config=None, **kwargs):
        set_seed(0)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        self.threshold = 50
        self.n_sections = 20
        self.n_orientations = 1
        self.n_latents = 8
        self.img_size = (256, 256)
        self.regions_masks = self.get_regions_masks()

        checkpoint_filepath = os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_fourier_spectrum_descriptors_model.pickle')
        self.pca_model = PCAModel.load_model(checkpoint_filepath)

        if self.config.set_BC_range:
            # computed on external reference dataset of 20 000 images -> np.percentile(0.01 - 0.99)
            range = np.load(os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_fourier_spectrum_descriptors_range.npz'))
            self.BC_range = [range['low'], range['high']]
        else:
            self.BC_range = [np.zeros(self.n_latents), np.ones(self.n_latents)]

        return

    def get_regions_masks(self):
        regions_masks = []
        # create sectors
        R = self.img_size[0] // 2
        section_regions = [(ring_idx / self.n_sections * R, (ring_idx + 1) / self.n_sections * R) for ring_idx in
                           range(self.n_sections)]

        # concatenate first and last regions
        orientation_regions = [(-np.pi / 2, -np.pi / 2 + 1 / self.n_orientations * np.pi / 2)]
        offset = -np.pi / 2 + 1 / self.n_orientations * np.pi / 2
        orientation_regions += [
            (offset + wedge_idx / self.n_orientations * np.pi, offset + (wedge_idx + 1) / self.n_orientations * np.pi)
            for
            wedge_idx in range(self.n_orientations - 1)]

        grid_x, grid_y = torch.meshgrid(torch.range(0, R - 1, 1), torch.range(-R, R - 1, 1))
        grid_r = (grid_x ** 2 + grid_y ** 2).sqrt()
        grid_theta = torch.atan(grid_y / grid_x)

        # fill feature vector
        for section_region in section_regions:
            r1 = section_region[0]
            r2 = section_region[1]

            # edge region
            theta1 = orientation_regions[0][0]
            theta2 = orientation_regions[0][1]

            region_mask = (grid_r >= r1) & (grid_r < r2) & (((grid_theta >= theta1) & (grid_theta < theta2)) | (
                        (grid_theta >= -offset) & (grid_theta <= np.pi / 2)))
            regions_masks.append(region_mask)

            # inner region
            for orientation_region in orientation_regions[1:]:
                theta1 = orientation_region[0]
                theta2 = orientation_region[1]

                region_mask = (grid_r >= r1) & (grid_r < r2) & (grid_theta >= theta1) & (grid_theta < theta2)
                regions_masks.append(region_mask)

        return regions_masks


    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                      for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                      for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)


    def spectrum_fourier_descriptors(self, image):
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        # power spectrum
        spectrum = torch.rfft(image, signal_ndim=2, onesided=False, normalized=True)
        power_spectrum = (spectrum[:, :, 0] ** 2 + spectrum[:, :, 1] ** 2)

        # shift to be centered
        power_spectrum = self.roll_n(power_spectrum, 1, power_spectrum.size(1) // 2)
        power_spectrum = self.roll_n(power_spectrum, 0, power_spectrum.size(0) // 2)

        # remove unrelevant frequencies
        power_spectrum[power_spectrum < power_spectrum.mean()] = 0
        half_power_spectrum = power_spectrum[power_spectrum.size(0) // 2:, :]

        # feature vector
        n_regions = self.n_sections * self.n_orientations
        feature_vector = torch.zeros(2 * n_regions)

        cur_region_idx = 0
        for region_mask in self.regions_masks:
            region_power_spectrum = deepcopy(half_power_spectrum)[region_mask]
            feature_vector[2 * cur_region_idx] = region_power_spectrum.mean()
            feature_vector[2 * cur_region_idx + 1] = region_power_spectrum.std()
            cur_region_idx += 1

        return feature_vector.cpu().numpy()

    def calc_embedding(self, x, **kwargs):
        # x: numpy H*W
        coefficients = self.spectrum_fourier_descriptors(x.cpu().squeeze().numpy())
        z = self.pca_model.calc_embedding(coefficients.reshape(1, -1)).squeeze()

        normalized_z = (z - self.BC_range[0]) / (self.BC_range[1] - self.BC_range[0])
        normalized_z = torch.from_numpy(normalized_z).unsqueeze(0)
        return normalized_z