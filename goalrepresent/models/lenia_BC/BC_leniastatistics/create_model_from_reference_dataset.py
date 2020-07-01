import numpy as np

import autodisc as ad
from autodisc.systems.lenia import LeniaStatistics
from goalrepresent.datasets import LENIADataset
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel

EPS = 0.0001


def calc_static_statistics(final_obs):
    '''Calculates the final statistics for lenia last observation'''

    feature_vector = np.zeros(17)
    cur_idx = 0

    size_y = final_obs.shape[0]
    size_x = final_obs.shape[1]
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
    distance_weight_matrix = LeniaStatistics.calc_distance_matrix(final_obs.shape[0],
                                                                  final_obs.shape[1])
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


if __name__ == '__main__':

    set_seed(0)
    n_features_BC = 8

    n_statistics = 17

    dataset_config = LENIADataset.default_config()
    dataset_config.data_root = '/gpfswork/rech/zaj/ucf28eq/data/lenia_datasets/data_005/'
    dataset_config.split = 'train'
    dataset = LENIADataset(config=dataset_config)

    # create fourier descriptors and save statistics
    coefficients = np.zeros((dataset.n_images, n_statistics))
    for idx in range(dataset.n_images):
        im = dataset.get_image(idx).squeeze().numpy()
        coeffs = calc_static_statistics(im)
        coefficients[idx] = coeffs


    # do PCA to keep only principal components according to reference dataset
    pca_model = PCAModel(n_features=n_statistics, n_latents=n_features_BC)
    X_all = coefficients.reshape(coefficients.shape[0], -1)
    z_all = pca_model.fit(X_all)
    pca_model.save_checkpoint('reference_dataset_pca_lenia_statistics_descriptors_model.pickle')

    np.savez('reference_dataset_pca_lenia_statistics_descriptors_range.npz',
             low=np.percentile(z_all, 0.01, axis=0), high=np.percentile(z_all, 99.9, axis=0))

    print('pca explained variance: {}'.format(pca_model.algorithm.explained_variance_ratio_.sum()))
    print(
        'analytic_space_range: {} - {}'.format(np.percentile(z_all, 0.01, axis=0), np.percentile(z_all, 99.9, axis=0)))

    np.savez('reference_dataset_pca_lenia_statistics_descriptors_statistics.npz',
             descriptors=coefficients, pca_explained_variance=pca_model.algorithm.explained_variance_ratio_)


    np.savez('reference_dataset_pca_lenia_statistics_descriptors_values.npz',z=z_all)




