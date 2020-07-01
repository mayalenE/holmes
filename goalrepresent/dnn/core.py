import os
import warnings

import torch
from torch import nn

import goalrepresent as gr
from goalrepresent.dnn.losses import losses
from goalrepresent.dnn.networks import encoders
from goalrepresent.dnn.solvers import initialization


class BaseDNN(nn.Module):
    """
    Base DNN Class
    Squeleton to follow for each dnn model, here simple single encoder that is not trained.
    """

    @staticmethod
    def default_config():
        default_config = gr.Config()

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # device parameters
        default_config.device = gr.Config()
        default_config.device.use_gpu = True

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = gr.Config()

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5

        # In training folder:
        ## logging (will save every X epochs)
        default_config.logging = gr.Config()
        default_config.logging.record_loss_every = 1
        default_config.logging.record_valid_images_every = 1
        default_config.logging.record_embeddings_every = 1

        ## checkpoints (will save model every X epochs)
        default_config.checkpoint = gr.Config()
        default_config.checkpoint.folder = None
        default_config.checkpoint.save_model_every = 10
        default_config.checkpoint.save_model_at_epochs = []

        ## evaluation (when we do testing during training, save every X epochs)
        default_config.evaluation = gr.Config()
        default_config.evaluation.folder = None
        default_config.evaluation.save_results_every = 1

        return default_config

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        # define the device to use (gpu or cpu)
        if self.config.device.use_gpu and not torch.cuda.is_available():
            self.config.device.use_gpu = False
            warnings.warn("Cannot set model device as GPU because not available, setting it to CPU")

        # network
        self.set_network(self.config.network.name, self.config.network.parameters)
        self.init_network(self.config.network.initialization.name, self.config.network.initialization.parameters)
        self.set_device(self.config.device.use_gpu)

        # loss function
        self.set_loss(self.config.loss.name, self.config.loss.parameters)

        # optimizer
        self.set_optimizer(self.config.optimizer.name, self.config.optimizer.parameters)

        self.n_epochs = 0

    def set_network(self, network_name, network_parameters):
        """
        Define the network modules, 
        Here simple encoder but this function is overwritten to include generator/discriminator.
        """
        self.network = nn.Module()
        encoder_class = encoders.get_encoder(network_name)
        self.network.encoder = encoder_class(config=network_parameters)

        # update config
        self.config.network.name = network_name
        self.config.network.parameters = gr.config.update_config(network_parameters, self.config.network.parameters)

    def init_network(self, initialization_name, initialization_parameters):
        initialization_function = initialization.get_initialization(initialization_name)
        if initialization_name == "pretrain":
            self.network = initialization_function(self.network, initialization_parameters.checkpoint_filepath)
        else:
            self.network.apply(initialization_function)

        # update config
        self.config.network.initialization.name = initialization_name
        self.config.network.initialization.parameters = gr.config.update_config(initialization_parameters,
                                                                                self.config.network.initialization.parameters)

    def set_device(self, use_gpu):
        if use_gpu:
            self.to("cuda:0")
        else:
            self.to("cpu")

        # update config
        self.config.device.use_gpu = use_gpu

    def push_variable_to_device(self, x):
        if next(self.parameters()).is_cuda and not x.is_cuda:
            x = x.cuda()
        return x

    def set_loss(self, loss_name, loss_parameters):
        loss_class = losses.get_loss(loss_name)
        self.loss_f = loss_class(**loss_parameters)

        # update config
        self.config.loss.name = loss_name
        self.config.loss.parameters = gr.config.update_config(loss_parameters, self.config.loss.parameters)

    def set_optimizer(self, optimizer_name, optimizer_parameters):
        optimizer_class = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer_class(self.network.parameters(),
                                         **optimizer_parameters)  # the optimizer acts on all the network nn.parameters by default

        # update config
        self.config.optimizer.name = optimizer_name
        self.config.optimizer.parameters = gr.config.update_config(optimizer_parameters,
                                                                   self.config.optimizer.parameters)

    def run_training(self, train_loader, n_epochs, valid_loader=None, training_logger=None):
        pass

    def train_epoch(self, train_loader, logger=None):
        self.train()
        pass

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        pass

    def save_checkpoint(self, checkpoint_filepath):
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
            "epoch": self.n_epochs,
            "type": self.__class__.__name__,
            "config": self.config,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(network, checkpoint_filepath)

    @staticmethod
    def load_checkpoint(checkpoint_filepath, use_gpu=False, representation_model=None):
        if os.path.exists(checkpoint_filepath):
            saved_model = torch.load(checkpoint_filepath, map_location='cpu')
            model_type = saved_model['type']
            config = saved_model['config']
            if not use_gpu:
                config.device.use_gpu = False
                if "ProgressiveTree" in model_type:
                    config.node.device.use_gpu = False
            # the saved dnn can belong to gr.dnn, gr.models, gr.evaluationmodels
            if hasattr(gr.dnn, model_type):
                model_cls = getattr(gr.dnn, model_type)
                model = model_cls(config=config)
            elif hasattr(gr.models, model_type):
                model_cls = getattr(gr.models, model_type)
                model = model_cls(config=config)
            elif hasattr(gr.evaluationmodels, model_type):
                model_cls = getattr(gr.evaluationmodels, model_type)
                model = model_cls(representation_model=representation_model, config=config)
            else:
                raise ValueError("the model cannot be load as it does not iherit from the BaseDNN class")

            if "ProgressiveTree" in model_type:
                split_history = saved_model['split_history']

                for split_node_path, split_node_attr in split_history.items():
                    model.split_node(split_node_path)
                    node = model.network.get_child_node(split_node_path)
                    node.boundary = split_node_attr["boundary"]
                    node.feature_range = split_node_attr["feature_range"]

            model.network.load_state_dict(saved_model['network_state_dict'])

            if "GAN" in model_type:
                model.optimizer_discriminator.load_state_dict(saved_model['optimizer_discriminator_state_dict'])
                model.optimizer_generator.load_state_dict(saved_model['optimizer_generator_state_dict'])
            else:
                model.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
            model.n_epochs = saved_model["epoch"]
            model.set_device(use_gpu)
            return model
        else:
            raise ValueError("checkpoint filepath does not exist")
