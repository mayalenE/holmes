import os

import numpy as np
import torch

import goalrepresent as gr
from goalrepresent.helper.randomhelper import set_seed
from .. import pytorchnnrepresentation


class BCBVAEModel():

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.set_BC_range = True
        return default_config

    def __init__(self, config=None, **kwargs):
        set_seed(0)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        # model
        config_initialization = gr.Config()
        config_initialization.type = 'load_pretrained_model'
        config_initialization.load_from_model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pth')
        self.load_model(config_initialization)
        self.n_latents = self.model.n_latents

        if self.config.set_BC_range:
            range = np.load( os.path.join(os.path.dirname(__file__), 'reference_dataset_bvae_descriptors_range.npz'))
            self.BC_range = [torch.from_numpy(range['low']).unsqueeze(0), torch.from_numpy(range['high']).unsqueeze(0)]
        else:
            self.BC_range = [torch.zeros(self.n_latents).unsqueeze(0), torch.ones(self.n_latents).unsqueeze(0)]

        if self.model.use_gpu:
            self.BC_range[0] = self.BC_range[0].cuda()
            self.BC_range[1] = self.BC_range[1].cuda()

        return

    def load_model(self, initialization=None):
        if initialization is None or 'type' not in initialization or initialization.type not in {'random_weight',
                                                                                                 'load_pretrained_model'}:
            initialization = gr.Config()
            initialization.type = 'random_weight'
            print('WARNING: wrong goal space initialization given so initializing with default Beta-VAE network')

        if initialization.type == 'random_weight':
            print('WARNING: Initializing random weight pytorch network for goal representation')
            # model type: 'AE', 'BetaVAE'
            if 'model_type' in initialization:
                model_type = initialization.model_type
            else:
                model_type = 'BetaVAE'
                print('WARNING: The model type is not specified so initializing Beta-VAE type network')
            try:
                model_cls = getattr(pytorchnnrepresentation, model_type)
            except:
                raise ValueError('Unknown initialization.model_type {!r}!'.format(initialization.model_type))
            # model init_params:
            if 'model_init_params' in initialization:
                model_init_params = initialization.model_init_params
            else:
                model_init_params = {'n_channels': 1, 'n_latents': 8, 'input_size': (256, 256), 'beta': 5.0,
                                     'use_gpu': True}
                print(
                    'WARNING: The model init params are not specified so initializing default network with parameters {0}'.format(
                        model_init_params))
            try:
                self.model = model_cls(**model_init_params)
            except:
                raise ValueError(
                    'Wrong initialization.model_init_params {!r}!'.format(initialization.model_init_params))


        elif initialization.type == 'load_pretrained_model':
            print('Initializing pre-trained pytorch network for goal representation')
            if 'load_from_model_path' in initialization:
                if os.path.exists(initialization.load_from_model_path):
                    saved_model = torch.load(initialization.load_from_model_path, map_location='cpu')
                    # model type: 'AE', 'BetaVAE'
                    if 'type' in saved_model:
                        model_type = saved_model['type']
                    else:
                        model_type = 'BetaVAE'
                        print('WARNING: The model type is not specified so initializing Beta-VAE type network')
                    try:
                        model_cls = getattr(pytorchnnrepresentation, model_type)
                        self.model_type = model_type
                    except:
                        raise ValueError('Unknown initialization.model_type {!r}!'.format(model_type))
                    # model init_params:
                    if 'init_params' in saved_model:
                        model_init_params = saved_model['init_params']
                    else:
                        model_init_params = {'n_channels': 1, 'n_latents': 8, 'input_size': (256, 256), 'beta': 5.0,
                                             'use_gpu': True}
                        print(
                            'WARNING: The model init params are not specified so initializing default network with parameters {0}'.format(
                                model_init_params))
                    try:
                        self.model = model_cls(**model_init_params)
                    except:
                        raise ValueError('Wrong initialization model_init_params {!r}!'.format(model_init_params))
                        # model state_dict:
                    try:
                        self.model.load_state_dict(saved_model['state_dict'])
                    except:
                        raise ValueError('Wrong state_dict of the loaded model')
                else:
                    raise ValueError('The model path {0} does not exist: cannot initialize network'.format(
                        initialization.load_from_model_path))
            else:
                raise ValueError(
                    'The network cannot be initalized because intialization config does not contain \'load_from_model_path\' parameter')

        # push model on gpu if available
        self.model.eval()
        if self.model.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()

        # initialize weights if specified
        if hasattr(initialization, 'initialize_weights'):
            self.set_model_weights(initialization.initialize_weights)

        return


    def preprocess(self, x):
        return x


    def calc_embedding(self, x, **kwargs):
        if isinstance(x, np.ndarray):
            # x: numpy H*W
            x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        else:
            x = x.float()
        self.model.eval()
        with torch.no_grad():
            z = self.model.calc(x)

        normalized_z = (z - self.BC_range[0]) / (self.BC_range[1] - self.BC_range[0])
        return normalized_z


# if __name__ == '__main__':
#     bc_bvae = BCBVAEModel()
#     for i in range(10):
#         x = np.random.rand(256, 256)
#         z = bc_bvae.calc_embedding(x)
#         print(z.shape, z.min(), z.max())