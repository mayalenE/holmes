import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

import goalrepresent as gr
from goalrepresent import dnn
from goalrepresent.dnn.networks import decoders
from goalrepresent.helper import tensorboardhelper

""" ========================================================================================================================
Base VAE architecture
========================================================================================================================="""
class VAEModel(dnn.BaseDNN, gr.BaseModel):
    '''
    Base VAE Class
    '''

    @staticmethod
    def default_config():
        default_config = dnn.BaseDNN.default_config()

        # network parameters
        default_config.network = gr.Config()
        default_config.network.name = "Burgess"
        default_config.network.parameters = gr.Config()
        default_config.network.parameters.n_channels = 1
        default_config.network.parameters.input_size = (64, 64)
        default_config.network.parameters.n_latents = 10
        default_config.network.parameters.n_conv_layers = 4
        default_config.network.parameters.feature_layer = 2
        default_config.network.parameters.encoder_conditional_type = "gaussian"

        # initialization parameters
        default_config.network.initialization = gr.Config()
        default_config.network.initialization.name = "pytorch"
        default_config.network.initialization.parameters = gr.Config()

        # loss parameters
        default_config.loss = gr.Config()
        default_config.loss.name = "VAE"
        default_config.loss.parameters = gr.Config()
        default_config.loss.parameters.reconstruction_dist = "bernoulli"

        # optimizer parameters
        default_config.optimizer = gr.Config()
        default_config.optimizer.name = "Adam"
        default_config.optimizer.parameters = gr.Config()
        default_config.optimizer.parameters.lr = 1e-3
        default_config.optimizer.parameters.weight_decay = 1e-5
        return default_config

    def __init__(self, config=None, **kwargs):
        dnn.BaseDNN.__init__(self, config=config, **kwargs)  # calls all constructors up to BaseDNN (MRO)

        self.output_keys_list = self.network.encoder.output_keys_list + ["recon_x"]

    def set_network(self, network_name, network_parameters):
        dnn.BaseDNN.set_network(self, network_name, network_parameters)
        # add a decoder to the network for the VAE
        decoder_class = decoders.get_decoder(network_name)
        self.network.decoder = decoder_class(config=network_parameters)

    def forward_from_encoder(self, encoder_outputs):
        decoder_outputs = self.network.decoder(encoder_outputs["z"])
        model_outputs = encoder_outputs
        model_outputs.update(decoder_outputs)
        return model_outputs

    def forward(self, x):
        if torch._C._get_tracing_state():
            return self.forward_for_graph_tracing(x)

        x = self.push_variable_to_device(x)
        encoder_outputs = self.network.encoder(x)
        return self.forward_from_encoder(encoder_outputs)

    def forward_for_graph_tracing(self, x):
        x = self.push_variable_to_device(x)
        z, feature_map = self.network.encoder.forward_for_graph_tracing(x)
        recon_x = self.network.decoder(z)
        return recon_x

    def calc_embedding(self, x, **kwargs):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        x = self.push_variable_to_device(x)
        self.eval()
        with torch.no_grad():
            z = self.network.encoder.calc_embedding(x)
        return z

    def run_training(self, train_loader, training_config, valid_loader=None, logger=None):
        """
        logger: tensorboard X summary writer
        """
        if "n_epochs" not in training_config:
            training_config.n_epochs = 0

        # Save the graph in the logger
        if logger is not None:
            dummy_input = torch.FloatTensor(1, self.config.network.parameters.n_channels,
                                            self.config.network.parameters.input_size[0],
                                            self.config.network.parameters.input_size[1]).uniform_(0, 1)
            dummy_input = self.push_variable_to_device(dummy_input)
            self.eval()
            with torch.no_grad():
                logger.add_graph(self, dummy_input, verbose=False)

        do_validation = False
        if valid_loader is not None:
            best_valid_loss = sys.float_info.max
            do_validation = True

        for epoch in range(training_config.n_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, logger=logger)
            t1 = time.time()

            if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                for k, v in train_losses.items():
                    logger.add_scalars('loss/{}'.format(k), {'train': v}, self.n_epochs)
                logger.add_text('time/train', 'Train Epoch {}: {:.3f} secs'.format(self.n_epochs, t1 - t0),
                                self.n_epochs)

            if self.n_epochs % self.config.checkpoint.save_model_every == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'current_weight_model.pth'))
            if self.n_epochs in self.config.checkpoint.save_model_at_epochs:
                self.save_checkpoint(os.path.join(self.config.checkpoint.folder, "epoch_{}_weight_model.pth".format(self.n_epochs)))

            if do_validation:
                t2 = time.time()
                valid_losses = self.valid_epoch(valid_loader, logger=logger)
                t3 = time.time()
                if logger is not None and (self.n_epochs % self.config.logging.record_loss_every == 0):
                    for k, v in valid_losses.items():
                        logger.add_scalars('loss/{}'.format(k), {'valid': v}, self.n_epochs)
                    logger.add_text('time/valid', 'Valid Epoch {}: {:.3f} secs'.format(self.n_epochs, t3 - t2),
                                    self.n_epochs)

                valid_loss = valid_losses['total']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(os.path.join(self.config.checkpoint.folder, 'best_weight_model.pth'))

    def train_epoch(self, train_loader, logger=None):
        self.train()
        losses = {}
        for data in train_loader:
            x = Variable(data['obs'])
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            # backward
            loss = batch_losses['total']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())

        for k, v in losses.items():
            losses[k] = np.mean(v)

        self.n_epochs += 1

        return losses

    def valid_epoch(self, valid_loader, logger=None):
        self.eval()
        losses = {}

        # Prepare logging
        record_valid_images = False
        record_embeddings = False
        if logger is not None:
            if self.n_epochs % self.config.logging.record_valid_images_every == 0:
                record_valid_images = True
                images = []
                recon_images = []
            if self.n_epochs % self.config.logging.record_embeddings_every == 0:
                record_embeddings = True
                embeddings = []
                labels = []
                if images is None:
                    images = []
        with torch.no_grad():
            for data in valid_loader:
                x = Variable(data['obs'])
                x = self.push_variable_to_device(x)
                # forward
                model_outputs = self.forward(x)
                loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
                batch_losses = self.loss_f(loss_inputs)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = np.expand_dims(v.detach().cpu().numpy(), axis=-1)
                    else:
                        losses[k] = np.vstack([losses[k], np.expand_dims(v.detach().cpu().numpy(), axis=-1)])

                if record_valid_images:
                    recon_x = model_outputs["recon_x"]
                    images.append(x)
                    recon_images.append(recon_x)

                if record_embeddings:
                    embeddings.append(model_outputs["z"])
                    labels.append(data["label"])
                    if not record_valid_images:
                        images.append(x)

        if record_valid_images:
            recon_images = torch.cat(recon_images)
            images = torch.cat(images)
        if record_embeddings:
            embeddings = torch.cat(embeddings)
            labels = torch.cat(labels)
            if not record_valid_images:
                images = torch.cat(images)

        # log results
        if record_valid_images:
            n_images = min(len(images), 40)
            sampled_ids = np.random.choice(len(images), n_images, replace=False)
            input_images = images[sampled_ids].detach().cpu()
            output_images = recon_images[sampled_ids].detach().cpu()
            if self.config.loss.parameters.reconstruction_dist == "bernoulli":
                output_images = torch.sigmoid(output_images)
            vizu_tensor_list = [None] * (2 * n_images)
            vizu_tensor_list[0::2] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[1::2] = [output_images[n] for n in range(n_images)]
            img = make_grid(vizu_tensor_list, nrow=2, padding=0)
            logger.add_image("reconstructions", img, self.n_epochs)

        if record_embeddings:
            images = tensorboardhelper.resize_embeddings(images)
            logger.add_embedding(
                embeddings,
                metadata=labels,
                label_img=images,
                global_step=self.n_epochs)

        # average loss and return
        for k, v in losses.items():
            losses[k] = np.mean(v)

        return losses

    def get_encoder(self):
        return deepcopy(self.network.encoder)

    def get_decoder(self):
        return deepcopy(self.network.decoder)


""" ========================================================================================================================
State-of-the-art modifications of the basic VAE
========================================================================================================================="""


class BetaVAEModel(VAEModel):
    '''
    BetaVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = VAEModel.default_config()

        # loss parameters
        default_config.loss.name = "BetaVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.beta = 5.0

        return default_config

    def __init__(self, config=None, **kwargs):
        VAEModel.__init__(self, config, **kwargs)


class AnnealedVAEModel(VAEModel):
    '''
    AnnealedVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = VAEModel.default_config()

        # loss parameters
        default_config.loss.name = "AnnealedVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.gamma = 1000.0
        default_config.loss.parameters.c_min = 0.0
        default_config.loss.parameters.c_max = 5.0
        default_config.loss.parameters.c_change_duration = 100000

        return default_config

    def __init__(self, config=None, **kwargs):
        VAEModel.__init__(self, config, **kwargs)

class BetaTCVAEModel(VAEModel):
    '''
    BetaTCVAE Class
    '''

    @staticmethod
    def default_config():
        default_config = VAEModel.default_config()

        # loss parameters
        default_config.loss.name = "BetaTCVAE"
        default_config.loss.parameters.reconstruction_dist = "bernoulli"
        default_config.loss.parameters.alpha = 1.0
        default_config.loss.parameters.beta = 10.0
        default_config.loss.parameters.gamma = 1.0
        default_config.loss.parameters.tc_approximate = 'mss'
        default_config.loss.parameters.dataset_size = 0

        return default_config

    def __init__(self, config=None, **kwargs):
        VAEModel.__init__(self, config, **kwargs)

    def train_epoch(self, train_loader, logger=None):
        self.train()
        losses = {}

        # update dataset size
        self.loss_f.dataset_size = len(train_loader.dataset)

        for data in train_loader:
            x = Variable(data['obs'])
            x = self.push_variable_to_device(x)
            # forward
            model_outputs = self.forward(x)
            loss_inputs = {key: model_outputs[key] for key in self.loss_f.input_keys_list}
            batch_losses = self.loss_f(loss_inputs)
            # backward
            loss = batch_losses['total']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())

        for k, v in losses.items():
            losses[k] = np.mean(v)

        self.n_epochs += 1

        return losses