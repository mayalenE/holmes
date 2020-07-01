import os

import numpy as np
import torch

import autodisc as ad
import goalrepresent as gr
from autodisc.systems.lenia import LeniaStatistics
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel

EPS = 0.0001

class BCLeniaStatisticsModel():

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.set_BC_range = True
        return default_config

    def __init__(self, config=None, **kwargs):
        set_seed(0)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        # model
        self.statistic_names = ['activation_mass', 'activation_volume',
                                        'activation_density', 'activation_mass_distribution',
                                        'activation_hu1', 'activation_hu2',
                                        'activation_hu3', 'activation_hu4',
                                        'activation_hu5', 'activation_hu6',
                                        'activation_hu7', 'activation_hu8',
                                        'activation_flusser9', 'activation_flusser10',
                                        'activation_flusser11', 'activation_flusser12',
                                        'activation_flusser13'
                                        ]
        self.n_statistics = len(self.statistic_names)
        self.n_latents = 8
        self.img_size = (256, 256)

        checkpoint_filepath = os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_lenia_statistics_descriptors_model.pickle')
        self.pca_model = PCAModel.load_model(checkpoint_filepath)

        if self.config.set_BC_range:
            # computed on external reference dataset of 20 000 images -> np.percentile(0.01 - 0.99)
            range = np.load( os.path.join(os.path.dirname(__file__), 'reference_dataset_pca_lenia_statistics_descriptors_range.npz'))
            self.BC_range = [range['low'], range['high']]
        else:
            self.BC_range = [np.zeros(self.n_latents), np.ones(self.n_latents)]

        return

    def calc_static_statistics(self, final_obs):
        '''Calculates the final statistics for lenia last observation'''

        feature_vector = np.zeros(self.n_statistics)
        cur_idx = 0

        size_y = self.img_size[0]
        size_x = self.img_size[1]
        num_of_cells = size_y * size_x

        # calc initial center of mass and use it as a reference point to "center" the world around it
        mid_y = (size_y - 1) / 2
        mid_x = (size_x - 1) / 2
        mid = np.array([mid_y, mid_x])

        activation_center_of_mass = np.array(LeniaStatistics.center_of_mass(final_obs))
        activation_shift_to_center = mid - activation_center_of_mass


        activation = final_obs
        centered_activation = np.roll(activation, activation_shift_to_center.astype(int), (0, 1))

        # calculate the image moments
        activation_moments = ad.helper.statistics.calc_image_moments(centered_activation)

        # activation mass
        activation_mass = activation_moments.m00
        activation_mass_data = activation_mass / num_of_cells  # activation is number of acitvated cells divided by the number of cells
        feature_vector[cur_idx] = activation_mass_data
        cur_idx += 1

        # activation volume
        activation_volume = np.sum(activation > EPS)
        activation_volume_data = activation_volume / num_of_cells
        feature_vector[cur_idx] = activation_volume_data
        cur_idx += 1

        # activation density
        if activation_volume == 0:
            activation_density_data = 0
        else:
            activation_density_data = activation_mass / activation_volume
        feature_vector[cur_idx] = activation_density_data
        cur_idx += 1

        # mass distribution around the center
        distance_weight_matrix = LeniaStatistics.calc_distance_matrix(self.img_size[0],
                                                                      self.img_size[1])
        if activation_mass <= EPS:
            activation_mass_distribution = 1.0
        else:
            activation_mass_distribution = np.sum(distance_weight_matrix * centered_activation) / np.sum(
                centered_activation)

        activation_mass_distribution_data = activation_mass_distribution
        feature_vector[cur_idx] = activation_mass_distribution_data
        cur_idx += 1

        # activation moments
        activation_hu1_data = activation_moments.hu1
        feature_vector[cur_idx] = activation_hu1_data
        cur_idx += 1

        activation_hu2_data = activation_moments.hu2
        feature_vector[cur_idx] = activation_hu2_data
        cur_idx += 1

        activation_hu3_data = activation_moments.hu3
        feature_vector[cur_idx] = activation_hu3_data
        cur_idx += 1

        activation_hu4_data= activation_moments.hu4
        feature_vector[cur_idx] = activation_hu4_data
        cur_idx += 1

        activation_hu5_data = activation_moments.hu5
        feature_vector[cur_idx] = activation_hu5_data
        cur_idx += 1

        activation_hu6_data = activation_moments.hu6
        feature_vector[cur_idx] = activation_hu6_data
        cur_idx += 1

        activation_hu7_data = activation_moments.hu7
        feature_vector[cur_idx] = activation_hu7_data
        cur_idx += 1

        activation_hu8_data = activation_moments.hu8
        feature_vector[cur_idx] = activation_hu8_data
        cur_idx += 1

        activation_flusser9_data = activation_moments.flusser9
        feature_vector[cur_idx] = activation_flusser9_data
        cur_idx += 1

        activation_flusser10_data = activation_moments.flusser10
        feature_vector[cur_idx] = activation_flusser10_data
        cur_idx += 1

        activation_flusser11_data = activation_moments.flusser11
        feature_vector[cur_idx] = activation_flusser11_data
        cur_idx += 1

        activation_flusser12_data = activation_moments.flusser12
        feature_vector[cur_idx] = activation_flusser12_data
        cur_idx += 1

        activation_flusser13_data = activation_moments.flusser13
        feature_vector[cur_idx] = activation_flusser13_data
        cur_idx += 1

        return feature_vector

    def calc_embedding(self, x, **kwargs):
        # x: numpy H*W
        statistic_coefficients = self.calc_static_statistics(x.cpu().squeeze().numpy())
        z = self.pca_model.calc_embedding(statistic_coefficients.reshape(1, -1)).squeeze()
        normalized_z = (z - self.BC_range[0]) / (self.BC_range[1] - self.BC_range[0])
        normalized_z = torch.from_numpy(normalized_z).unsqueeze(0)
        return normalized_z


# if __name__ == '__main__':
#     bc_leniastatistics = BCLeniaStatisticsModel()
#     for i in range(10):
#         x = torch.zeros((256, 256)).uniform_()
#         z = bc_leniastatistics.calc_embedding(x)
#         print(z)