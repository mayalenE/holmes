import os
import re
from glob import glob

import numpy as np
import torch

import autodisc as ad
import goalrepresent as gr
from goalrepresent.datasets.image.imagedataset import LENIADataset

"""=====================================================================================================================
# Measure of similarity from paper "Similarity of Neural Network Representations Revisited"
# code: https://github.com/google-research/google-research/tree/master/representation_similarity
====================================================================================================================="""
def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)

def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)

"""====================================================================================================================="""

def collect_representations(model, test_dataset):
    data = []

    if hasattr(model, "eval"):
        model.eval()


    with torch.no_grad():

        for test_data_idx in range(test_dataset.n_images):
            x = test_dataset.__getitem__(test_data_idx)["obs"].unsqueeze(0)
            z = model.calc_embedding(x).squeeze().detach().cpu().numpy()
            data.append(z)


        data = np.stack(data, axis=0)



    return data


if __name__ == '__main__':

    # Load external dataset of 3000 images (half SLP and half TLP)
    test_dataset_config = ad.Config()
    test_dataset_config.data_root = "/gpfswork/rech/zaj/ucf28eq/data/lenia_datasets/data_005/"
    test_dataset_config.split = "test"
    test_dataset = LENIADataset(config=test_dataset_config)
    # uncomment following lines for debugging
    # test_dataset.n_images = 20
    # test_dataset.labels = torch.cat([test_dataset.labels[:10], test_dataset.labels[-10:]])
    # test_dataset.images = torch.cat([test_dataset.images[:10], test_dataset.images[-10:]])



    # Load checkpoints per training stage
    checkpoint_filepath = "training/checkpoints/*.pth"
    checkpoint_matches = glob(checkpoint_filepath)

    staged_models = {}
    for match in checkpoint_matches:
        if "stage_" not in match:
            continue;
        id_as_str = re.findall('_(\d+).', match)
        if len(id_as_str) > 0:
            cur_stage_idx = int(id_as_str[-1]) # use the last find, because ther could be more number in the filepath, such as in a directory name
            staged_models[cur_stage_idx] = gr.dnn.BaseDNN.load_checkpoint(match, use_gpu=False)

    n_stages = len(staged_models)
    CKA = np.zeros((n_stages, n_stages))

    representations_per_stage = []
    for i in range(n_stages):
        representations_per_stage.append(collect_representations(staged_models[i], test_dataset))# N x d (N=N_images, d=n_latents)

    for i in range(n_stages):
        for j in range(i, n_stages):
            CKA_ij = feature_space_linear_cka(representations_per_stage[i], representations_per_stage[j])
            CKA[i,j] = CKA_ij
            CKA[j,i] = CKA_ij

    statistic_foler = './statistics'
    np.save(os.path.join(statistic_foler, 'temporal_RSA.npy'), CKA)
    np.save(os.path.join(statistic_foler, 'test_dataset_representations_per_stage.npy'), np.stack(representations_per_stage))




