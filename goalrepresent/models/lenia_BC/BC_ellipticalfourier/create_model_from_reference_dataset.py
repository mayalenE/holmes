from copy import deepcopy

import cv2
import numpy as np
import pyefd
import torch

from goalrepresent.datasets.image.imagedataset import LENIADataset
from goalrepresent.helper.randomhelper import set_seed
from goalrepresent.models import PCAModel


def get_contour(image, threshold=50):
    image_8bit = np.uint8(image * 255)

    threshold_level = threshold
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


def calc_contours_distance(contour_ref, contour_2):
    dist = 0
    contour_ref = contour_ref.astype('float')
    contour_2_aligned = np.zeros_like(contour_ref)
    for point_ref_idx in range(len(contour_ref)):
        point_ref = contour_ref[point_ref_idx]
        closest_point = None
        closest_point_dist = 500
        for point_2 in contour_2:
            cur_dist = np.sqrt(((point_ref - point_2) ** 2).sum())
            if cur_dist < closest_point_dist:
                closest_point = point_2
                closest_point_dist = cur_dist
        contour_2_aligned[point_ref_idx] = closest_point

    dist = np.max(np.sqrt(((contour - contour_2_aligned) ** 2).sum(axis=1)))

    return dist


if __name__ == '__main__':

    set_seed(0)
    n_features_BC = 8

    threshold = 50
    n_harmonics = 25

    dataset_config = LENIADataset.default_config()
    dataset_config.data_root = '/gpfswork/rech/zaj/ucf28eq/data/lenia_datasets/data_005/'
    dataset_config.split = 'train'
    dataset = LENIADataset(config=dataset_config)
    animal_ids = torch.where(dataset.labels == 0)[0].numpy()

    # create fourier descriptors and save statistics
    normalized_coefficients = np.zeros((dataset.n_images, n_harmonics, 4))
    errors = np.zeros(dataset.n_images)
    for idx in range(dataset.n_images):
        im = dataset.get_image(idx).squeeze().numpy()
        contour = get_contour(im, threshold=threshold)
        if contour is not None:
            # elliptical Fourier descriptor's coefficients.
            coeffs = pyefd.elliptic_fourier_descriptors(contour, order=n_harmonics)
            # normalize coeffs
            normalized_coeffs, L = pyefd.normalize_efd(deepcopy(coeffs))
            # recon_contour
            if len(contour) > 0:
                locus = pyefd.calculate_dc_coefficients(contour)
            else:
                locus = (0, 0)
            recon_contour = pyefd.reconstruct_contour(coeffs, locus=locus, num_points=300)
            # error measure
            error = calc_contours_distance(contour, recon_contour) / (2 * L) * 100
            normalized_coefficients[idx] = normalized_coeffs
            errors[idx] = error


    print('{} harmonics:  error {}(mean); {}(std); {} (min); {}(max); {}(90p)'.format(n_harmonics, errors.mean(),
                                                                                      errors.std(), errors.min(),
                                                                                      errors.max(),
                                                                                      np.percentile(errors, 90)))

    # do PCA to keep only principal components according to reference dataset
    pca_model = PCAModel(n_features=n_harmonics * 4, n_latents=n_features_BC)
    X_animal = normalized_coefficients[animal_ids].reshape(normalized_coefficients[animal_ids].shape[0], -1)
    z_animal = pca_model.fit(X_animal)
    pca_model.save_checkpoint('reference_dataset_pca_fourier_elliptical_descriptors_model.pickle')

    np.savez('reference_dataset_pca_fourier_elliptical_descriptors_range.npz',
             low=np.percentile(z_animal, 0.01, axis=0), high=np.percentile(z_animal, 99.9, axis=0))

    print('pca explained variance: {}'.format(pca_model.algorithm.explained_variance_ratio_.sum()))
    print(
        'analytic_space_range: {} - {}'.format(np.percentile(z_animal, 0.01, axis=0), np.percentile(z_animal, 99.9, axis=0)))

    np.savez('reference_dataset_pca_fourier_elliptical_descriptors_statistics.npz',
             descriptors=normalized_coefficients, contour_reconstruction_error=errors,
             pca_explained_variance=pca_model.algorithm.explained_variance_ratio_, animal_ids=animal_ids)

    X_all = normalized_coefficients.reshape(normalized_coefficients.shape[0], -1)
    z_all = pca_model.calc_embedding(X_all)
    np.savez('reference_dataset_pca_fourier_elliptical_descriptors_values.npz',
             z=z_all)




