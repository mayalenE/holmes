import os
from copy import deepcopy

import cv2
import numpy as np
import torch

import goalrepresent as gr
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel
from . import pyefd


class BCEllipticalFourierModel():

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.set_BC_range = True
        return default_config

    def __init__(self, config=None, **kwargs):
        set_seed(0)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        self.threshold = 50
        self.n_harmonics = 25
        self.n_latents = 8

        checkpoint_filepath = os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_fourier_elliptical_descriptors_model.pickle')
        self.pca_model = PCAModel.load_model(checkpoint_filepath)

        if self.config.set_BC_range:
            # computed on external reference dataset of 20 000 images -> np.percentile(0.01 - 0.99)
            range = np.load(os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_fourier_elliptical_descriptors_range.npz'))
            self.BC_range = [range['low'], range['high']]
        else:
            self.BC_range = [np.zeros(self.n_latents), np.ones(self.n_latents)]

        return

    def get_contour(self, image):
        image_8bit = np.uint8(image * 255)

        threshold_level = self.threshold
        _, binarized = cv2.threshold(image_8bit, threshold_level, 255, cv2.THRESH_BINARY)

        # Find the contours of a binary image using OpenCV.
        _, contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # we keep only the longest countour
        main_contour = None
        main_contour_T = 0
        for cont in contours:
            dxy = np.diff(cont, axis=0)
            dt = np.sqrt((dxy ** 2).sum(axis=1))
            t = np.concatenate([([0.]), np.cumsum(dt)])
            T = t[-1]
            if T > main_contour_T:
                main_contour = cont
                main_contour_T = T

        if main_contour is not None:
            main_contour = main_contour.squeeze()

        return main_contour

    def calc_embedding(self, x, **kwargs):
        contour = self.get_contour(x.cpu().squeeze().numpy())
        z = np.zeros(self.n_latents)
        if contour is not None:
            # elliptical Fourier descriptor's coefficients.
            coeffs = pyefd.elliptic_fourier_descriptors(contour, order=self.n_harmonics)
            # normalize coeffs
            normalized_coeffs, _ = pyefd.normalize_efd(deepcopy(coeffs))
            z = self.pca_model.calc_embedding(normalized_coeffs.reshape(1,-1)).squeeze()

        normalized_z = (z - self.BC_range[0]) / (self.BC_range[1] - self.BC_range[0])
        normalized_z = torch.from_numpy(normalized_z).unsqueeze(0)
        return normalized_z

# if __name__ == '__main__':
#     bc_ellipticalfourier = BCEllipticalFourierModel()
#     for i in range(10):
#         x = np.random.rand(256, 256)
#         z = bc_ellipticalfourier.calc_embedding(x)
#         print(z.shape, z.min(), z.max())