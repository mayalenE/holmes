import pickle

import numpy as np
from sklearn.decomposition import PCA

import goalrepresent as gr


class PCAModel(gr.BaseModel):
    '''
    PCA Model Class
    '''

    @staticmethod
    def default_config():
        default_config = gr.BaseModel.default_config()

        # hyperparameters
        default_config.hyperparameters = gr.Config()
        default_config.hyperparameters.random_state = None

        return default_config

    def __init__(self, n_features=28 * 28, n_latents=10, config=None, **kwargs):
        gr.BaseModel.__init__(self, config=config, **kwargs)

        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']

        # input size (flatten)
        self.n_features = n_features
        # latent size
        self.n_latents = n_latents
        # feature range
        self.feature_range = (0.0, 1.0)

        self.algorithm = PCA()
        self.update_algorithm_parameters()

    def fit(self, X_train, update_range=True):
        ''' 
        X_train: array-like (n_samples, n_features)
        '''
        X_train = np.nan_to_num(X_train)
        if update_range:
            self.feature_range = (X_train.min(axis=0), X_train.max(axis=0))  # save (min, max) for normalization
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(
            scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        X_train = (X_train - self.feature_range[0]) / scale
        X_transformed = self.algorithm.fit_transform(X_train)
        return X_transformed

    def calc_embedding(self, x):
        scale = self.feature_range[1] - self.feature_range[0]
        scale[np.where(
            scale == 0)] = 1.0  # trick when some some latents are the same for every point (no scale and divide by 1)
        x = (x - self.feature_range[0]) / scale
        x = self.algorithm.transform(x)
        return x

    def update_hyperparameters(self, hyperparameters):
        gr.BaseModel.update_hyperparameters(self, hyperparameters)
        self.update_algorithm_parameters()

    def update_algorithm_parameters(self):
        self.algorithm.set_params(n_components=self.n_latents, **self.config.hyperparameters)

    def get_encoder(self):
        return

    def get_decoder(self):
        return

    def save_checkpoint(self, checkpoint_filepath):
        with open(checkpoint_filepath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def load_model(checkpoint_filepath):
        with open(checkpoint_filepath, 'rb') as f:
            pca_model = pickle.load(f)
        return pca_model
