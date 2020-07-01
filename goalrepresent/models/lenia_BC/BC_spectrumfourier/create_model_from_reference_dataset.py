from copy import deepcopy

import numpy as np
import torch

from goalrepresent.datasets.image.imagedataset import LENIADataset
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel


def complex2polar(x, y):
    r = (x ** 2 + y ** 2).sqrt()
    theta = torch.atan(y / x)
    return r, theta

def polar2complex(r, theta):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x,y

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def spectrum_fourier_descriptors(image, n_sections=4, n_orientations=7):
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        image = image.cuda()

    # power spectrum
    spectrum = torch.rfft(image, signal_ndim=2, onesided=False, normalized=True)
    magnitude, phase = complex2polar(spectrum[:,:,0], spectrum[:,:,1])
    power_spectrum = magnitude.pow(2)

    # shift to be centered
    power_spectrum = roll_n(power_spectrum, 1, power_spectrum.size(1) // 2)
    power_spectrum = roll_n(power_spectrum, 0, power_spectrum.size(0) // 2)

    # remove unrelevant frequencies
    power_spectrum[power_spectrum<power_spectrum.mean()] = 0
    half_power_spectrum = power_spectrum[power_spectrum.size(0) // 2:, :]

    # prepare feature vector
    n_regions = n_sections * n_orientations
    feature_vector = torch.zeros(2*n_regions)

    # create sectors
    R = image.shape[0] // 2
    section_regions = [(ring_idx/n_sections*R, (ring_idx+1)/n_sections*R) for ring_idx in range(n_sections)]

    # concatenate first and last regions
    orientation_regions = [(-np.pi/2, -np.pi/2 + 1 / n_orientations * np.pi / 2 )]
    offset = -np.pi/2 + 1 / n_orientations * np.pi / 2
    orientation_regions += [(offset + wedge_idx / n_orientations * np.pi, offset+ (wedge_idx + 1) / n_orientations * np.pi) for wedge_idx in range(n_orientations - 1)]

    grid_x, grid_y = torch.meshgrid(torch.range(0, R - 1, 1), torch.range(-R, R - 1, 1))
    grid_r, grid_theta = complex2polar(grid_x, grid_y)


    # fill feature vector
    cur_region_idx = 0
    for section_region in section_regions:
        r1 = section_region[0]
        r2 = section_region[1]

        # edge region
        theta1 = orientation_regions[0][0]
        theta2 = orientation_regions[0][1]

        region_mask = (grid_r>=r1) & (grid_r<r2) & (((grid_theta>=theta1) & (grid_theta<theta2)) | ((grid_theta>=-offset) & (grid_theta<=np.pi/2)))
        region_power_spectrum = deepcopy(half_power_spectrum)[region_mask]
        feature_vector[2*cur_region_idx] = region_power_spectrum.mean()
        feature_vector[2*cur_region_idx+1] = region_power_spectrum.std()
        cur_region_idx += 1


        # inner region
        for orientation_region in orientation_regions[1:]:
            theta1 = orientation_region[0]
            theta2 = orientation_region[1]

            region_mask = (grid_r >= r1) & (grid_r < r2) & (grid_theta >= theta1) & (grid_theta < theta2)
            region_power_spectrum = deepcopy(half_power_spectrum)[region_mask]
            feature_vector[2 * cur_region_idx] = region_power_spectrum.mean()
            feature_vector[2 * cur_region_idx + 1] = region_power_spectrum.std()
            cur_region_idx += 1


    return feature_vector.detach().cpu().numpy()

if __name__ == '__main__':

    set_seed(0)
    n_features_BC = 8


    n_sections = 20
    n_orientations = 1

    dataset_config = LENIADataset.default_config()
    dataset_config.data_root = '/gpfswork/rech/zaj/ucf28eq/data/lenia_datasets/data_005/'
    dataset_config.split = 'train'
    dataset = LENIADataset(config=dataset_config)
    non_animal_ids = torch.where(dataset.labels == 1)[0].numpy()

    # create fourier descriptors and save statistics
    normalized_coefficients = np.zeros((dataset.n_images, 2*n_sections*n_orientations))
    for idx in range(dataset.n_images):
        im = dataset.get_image(idx).squeeze().numpy()
        # spectrum Fourier descriptor's coefficients.
        coeffs = spectrum_fourier_descriptors(im, n_sections=n_sections, n_orientations=n_orientations)
        normalized_coefficients[idx] = coeffs


    # do PCA to keep only principal components according to reference dataset
    pca_model = PCAModel(n_features=2*n_sections*n_orientations, n_latents=n_features_BC)
    X_non_animal = normalized_coefficients[non_animal_ids].reshape(normalized_coefficients[non_animal_ids].shape[0], -1)
    z_non_animal = pca_model.fit(X_non_animal)
    pca_model.save_checkpoint('reference_dataset_pca_fourier_spectrum_descriptors_model.pickle')

    np.savez('reference_dataset_pca_fourier_spectrum_descriptors_range.npz',
             low=np.percentile(z_non_animal, 0.01, axis=0), high=np.percentile(z_non_animal, 99.9, axis=0))

    print('pca explained variance: {}'.format(pca_model.algorithm.explained_variance_ratio_.sum()))
    print('analytic_space_range: {} - {}'.format(np.percentile(z_non_animal, 0.01, axis=0), np.percentile(z_non_animal, 99.9, axis=0)))

    np.savez('reference_dataset_pca_fourier_spectrum_descriptors_statistics.npz',
             descriptors=normalized_coefficients, pca_explained_variance=pca_model.algorithm.explained_variance_ratio_, non_animal_ids=non_animal_ids)

    X_all = normalized_coefficients.reshape(normalized_coefficients.shape[0], -1)
    z_all = pca_model.calc_embedding(X_all)
    np.savez('reference_dataset_pca_fourier_spectrum_descriptors_values.npz',
             z=z_all)






