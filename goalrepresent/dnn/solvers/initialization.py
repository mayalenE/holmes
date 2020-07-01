import math
import os

import torch
from torch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_, uniform_, eye_


def get_initialization(initialization_name):
    '''
    initialization_name: string such that the function called is weights_init_<initialization_name>
    '''
    initialization_name = initialization_name.lower()
    return eval("weights_init_{}".format(initialization_name))

def weights_init_pretrain(network, checkpoint_filepath):
    if os.path.exists(checkpoint_filepath):
        saved_model = torch.load (checkpoint_filepath, map_location='cpu')
        network.load_state_dict(saved_model['network_state_dict'])
    else:
        print("WARNING: the checkpoint filepath for a pretrain initialization has not been found, skipping the initialization")
    return network



def weights_init_null(m):
    """
    For HOLMES: initialize zero net (child born with no knowledge) and identity connections from parent (child start by copying parent)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_connections_identity(m):
    """
    For HOLMES: initialize identity connections
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(1) # for 1*1 convolution is equivalent to identity
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        eye_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_uniform(m, a=0., b=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        uniform_(m.weight.data, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        uniform_(m.weight.data, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_pytorch(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        m.reset_parameters()
        
def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def weights_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def weights_init_custom_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.uniform_(-1,1)
        m.weight.data.uniform_(-1/(m.weight.size(2)), 1/(m.weight.size(2))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-1/math.sqrt(m.weight.size(0)), 1/math.sqrt(m.weight.size(0))) 
        if m.bias is not None:
            m.bias.data.uniform_(-0.1,0.1)
            