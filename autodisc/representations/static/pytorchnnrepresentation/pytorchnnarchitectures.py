from autodisc.representations.static.pytorchnnrepresentation import helper
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import math
import os
from torchvision.utils import save_image
from autodisc.representations.static.pytorchnnrepresentation.models import encoders, decoders
EPS = 1e-12

""" ========================================================================================================================
Base VAE architecture
========================================================================================================================="""

class VAE(nn.Module):
    '''
    Base VAE Class
    '''
    def __init__(self, n_channels = 1, input_size = (64,64), n_latents = 10, model_architecture = "Burgess", n_conv_layers = 4, reconstruction_dist = 'bernouilli', use_gpu = True, **kwargs):
        super(VAE, self).__init__()
        
        # store the initial parameters used to create the model
        self.init_params = locals()
        del self.init_params['self']
        
        # define the device to use (gpu or cpu)
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = n_channels
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers
        self.n_latents = n_latents
        
        # network
        encoder = encoders.get_encoder(model_architecture)
        decoder = decoders.get_decoder(model_architecture)
        self.encoder = encoder(self.n_channels, self.input_size, self.n_conv_layers, self.n_latents)
        self.decoder = decoder(self.n_channels, self.input_size, self.n_conv_layers, self.n_latents)
        self.reconstruction_dist = reconstruction_dist
        
        self.n_epochs = 0
        
    def encode(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return self.encoder(x)
    
    def decode(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.decoder(z)

    def forward(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {'recon_x': self.decode(z), 'mu': mu, 'logvar': logvar, 'sampled_z': z}
    
    def reparameterize(self, mu, logvar):
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def calc(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        return mu
    
    def recon_loss (self, recon_x, x):
        if self.reconstruction_dist == "bernouilli":
            return helper.BCE_with_digits_loss(recon_x, x)  
        elif self.reconstruction_dist == "gaussian":
            return helper.MSE_loss(recon_x,x)
        else:
            raise ValueError ("Unkown decoder distribution: {}".format(self.reconstruction_dist))
            
    def train_loss(self, outputs, inputs):
        """ 
        train loss
        ------------
        recon_x: reconstructed images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        recon_loss = self.recon_loss(recon_x, x)
        KLD_loss, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total_loss = recon_loss + KLD_loss
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
    def valid_losses(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        BCE_loss = helper.BCE_with_digits_loss(recon_x, x) 
        MSE_loss = helper.MSE_loss(recon_x,x)
        if self.reconstruction_dist == "bernouilli":
            recon_loss = BCE_loss
        elif self.reconstruction_dist == "gaussian":
            recon_loss = MSE_loss
        KLD_loss, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total_loss = recon_loss + KLD_loss
        
        return {'total': total_loss, 'recon': recon_loss, 'BCE': BCE_loss, 'MSE': MSE_loss, 'KLD': KLD_loss, 'KLD_var': KLD_var}
    
    def set_optimizer(self, optimizer_name, optimizer_hyperparameters):
        optimizer = eval("torch.optim.{}".format(optimizer_name))
        self.optimizer = optimizer(self.parameters(), **optimizer_hyperparameters)
    
    def train_epoch (self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            input_img = Variable(data['image'])    
            # forward
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
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
            losses [k] = np.mean (v)
        
        self.n_epochs += 1
        return losses
    
    
    def valid_epoch (self, valid_loader, save_image_in_folder=None):
        self.eval()
        losses = {}
        with torch.no_grad():
            for data in valid_loader:
                input_img = Variable(data['image'])
                # forward
                outputs = self.forward(input_img)
                batch_losses = self.valid_losses(outputs, data)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                 #break
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
            
        # save images
        if save_image_in_folder is not None and self.n_epochs % 10 == 0:
            input_images = input_img.cpu().data
            if self.reconstruction_dist == 'bernouilli':
                output_images = torch.sigmoid(outputs['recon_x']).cpu().data
            else:
                output_images = outputs['recon_x'].cpu().data
            n_images = data['image'].size()[0]
            vizu_tensor_list = [None] * (2*n_images)
            vizu_tensor_list[:n_images] = [input_images[n] for n in range(n_images)]
            vizu_tensor_list[n_images:] = [output_images[n] for n in range(n_images)]
            filename = os.path.join (save_image_in_folder, 'Epoch{0}.png'.format(self.n_epochs))
            save_image(vizu_tensor_list, filename, nrow=n_images, padding=0)

        return losses
    
    def update_hyperparameters(self, hyperparameters):
        '''
        hyperparameters: dictionary of 'name': value (value should be a float)
        '''
        for hyperparam_key, hyperparam_val in hyperparameters.items():
            if hasattr(self, hyperparam_key):
                setattr(self, hyperparam_key, float(hyperparam_val))



""" ========================================================================================================================
State-of-the-art modifications of the basic VAE
========================================================================================================================="""

class BetaVAE(VAE):
    '''
    BetaVAE Class
    '''
    def __init__(self, beta=5.0, add_var_to_KLD_loss=False, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        # beta is scaled for 256x256 images so we resize it based on the input size
        crop_ratio = float(256 //self.input_size[0])
        self.beta = beta / crop_ratio
        self.add_var_to_KLD_loss = add_var_to_KLD_loss
    
    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        recon_loss = self.recon_loss(recon_x, x)
        KLD_loss, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total_loss = recon_loss + self.beta * (KLD_loss + float(self.add_var_to_KLD_loss) * KLD_var)
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss, 'KLD_var': KLD_var}
    
    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)
    
    
class AnnealedVAE(VAE):
    '''
    AnnealedVAE Class
    '''
    def __init__(self, gamma=1000.0, c_min=0.0, c_max=5.0, c_change_duration=100000, **kwargs):
        super(AnnealedVAE, self).__init__(**kwargs)
        
        self.gamma = gamma
        self.c_min = c_min
        self.c_max = c_max
        self.c_change_duration = c_change_duration
        
        self.n_iters = 0
    
    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        recon_loss = self.recon_loss(recon_x, x)
        KLD_loss, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total_loss = recon_loss + self.gamma * (KLD_loss - self.C).abs()
        return {'total': total_loss, 'recon': recon_loss, 'KLD': KLD_loss}
    
    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)
    
    def update_encoding_capacity(self):
        if self.n_iters > self.c_change_duration:
            self.C = self.c_max
        else:
            self.C =  min(self.c_min + (self.c_max - self.c_min) * self.n_iters / self.c_change_duration, self.c_max)
    
    def train_epoch (self, train_loader):
        self.train()
        losses = {}
        for data in train_loader:
            input_img = Variable(data['image'])    
            # update capacity
            self.n_iters += 1
            self.update_encoding_capacity()
            # forward
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
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
            losses [k] = np.mean (v)
        
        self.n_epochs +=1
        return losses
    
    
class BetaTCVAE(VAE):
    '''
    $\beta$-TCVAE Class
    '''
    def __init__(self, dataset_size=0, alpha = 1.0, beta = 10.0, gamma = 1.0, tc_approximate = 'mss', **kwargs):
        super(BetaTCVAE, self).__init__(**kwargs)
        
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tc_approximate = tc_approximate.lower()


    def train_loss(self, outputs, inputs):
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        sampled_z = outputs['sampled_z']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
            
        if self.use_gpu and not sampled_z.is_cuda:
            sampled_z = sampled_z.cuda()
        
        # RECON LOSS
        recon_loss = self.recon_loss(recon_x, x)
        
        # KL LOSS MODIFIED
        ## calculate log q(z|x) (log density of gaussian(mu,sigma2))
        log_q_zCx = (-0.5 * (math.log(2.0*np.pi) + logvar) - (sampled_z-mu).pow(2) / (2 * logvar.exp())).sum(1) # sum on the latent dimensions (factorized distribution so log of prod is sum of logs)
        
        ## calculate log p(z) (log density of gaussian(0,1))
        log_pz = (-0.5 * math.log(2.0*np.pi) - sampled_z.pow(2) / 2).sum(1)
        
        ## calculate log_qz ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m)) and log_prod_qzi
        batch_size = sampled_z.size(0)
        _logqz = -0.5 * (math.log(2.0*np.pi) + logvar.view(1, batch_size, self.n_latents)) - (sampled_z.view(batch_size, 1, self.n_latents) - mu.view(1, batch_size, self.n_latents)).pow(2) / (2 * logvar.view(1, batch_size, self.n_latents).exp())
        if self.tc_approximate == 'mws':
            # minibatch weighted sampling
            log_prod_qzi = (helper.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * self.dataset_size)).sum(1)
            log_qz = (helper.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.dataset_size))
        elif self.tc_approximate == 'mss':
            # minibatch stratified sampling
            N = self.dataset_size
            M = batch_size - 1
            strat_weight = (N - M) / (N * M)
            W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
            W.view(-1)[::M+1] = 1 / N
            W.view(-1)[1::M+1] = strat_weight
            W[M-1, 0] = strat_weight
            logiw_matrix = Variable(W.log().type_as(_logqz.data))
            log_qz = helper.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            log_prod_qzi = helper.logsumexp(logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)
        else:
            raise ValueError ('The minibatch approximation of the total correlation "{}" is not defined'.format(self.tc_approximate))

        ## I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        ## TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        ## dw_kl_loss is KL[q(z)||p(z)] (dimension-wise KL term)
        dw_kl_loss = (log_prod_qzi - log_pz).mean()


        # TOTAL LOSS
        total_loss = recon_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss
        return {'total': total_loss, 'recon': recon_loss, 'mi': mi_loss, 'tc': tc_loss, 'dw_kl': dw_kl_loss}
    
    def valid_losses(self, outputs, inputs):
        return self.train_loss(outputs, inputs)
    
    def train_epoch (self, train_loader):
        self.train()
        losses = {}
        self.dataset_size = len(train_loader.dataset)
        
        for data in train_loader:
            input_img = Variable(data['image'])    
            # forward
            outputs = self.forward(input_img)
            batch_losses = self.train_loss(outputs, data)
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
            #break
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)
        
        self.n_epochs +=1
        return losses