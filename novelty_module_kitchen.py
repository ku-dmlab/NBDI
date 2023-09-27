import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque

import os
import numpy as np
from PIL import Image
from torchvision.transforms import Resize

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque


class Phi(nn.Module):
    def __init__(self):
        super(Phi, self).__init__()
        
        # state : (128, 60) -> phi(st) : (128, 80)
        self.linear1 = nn.Linear(60,120)
        self.linear2 = nn.Linear(120,80)

    def forward(self,x):
        # x = F.normalize(x)
        y = F.elu(self.linear1(x))
        y = F.elu(self.linear2(y))
        # y = y.flatten(start_dim=1) 
        return y

class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        
        # phi(st) : (128, 80) + phi(st) : (128, 80)  = (128, 160)
        self.linear1 = nn.Linear(160,80)
        self.linear2 = nn.Linear(80,9)

    def forward(self, state1,state2):
        
        # phi(st) : (128, 288) + phi(st+1) : (128, 288) = (128, 576)
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        # y = F.softmax(y,dim=1)
        return y

class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        
        # phi(st) : (128, 80) + action : (128, 9) = (128, 89)
        # (128, 89) -> (128, 80)
        self.linear1 = nn.Linear(89,120)
        self.linear2 = nn.Linear(120,80)

    def forward(self,state,action):
        x = torch.cat( (state,action) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y

def loss_fn(inverse_loss, forward_loss):
    
    params = {
    'batch_size':150,
    'beta':0.2,
    'lambda':0.1,
    'eta': 1.0,
    'gamma':0.2,
    'max_episode_len':100,
    'min_progress':15,
    'action_repeats':6,
    'frames_per_state':3
} 
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    
    loss = loss_.sum() / loss_.flatten().shape[0]
    return loss


def ICM(state1, action, state2, forward_scale=1., inverse_scale=1e4):
    encoder = Phi().to("cuda")
    forward_model = Fnet().to("cuda")
    inverse_model = Gnet().to("cuda")
    
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
    
    forward_loss = nn.MSELoss(reduction='none').to("cuda")
    inverse_loss = nn.MSELoss(reduction='none').to("cuda")
    

    forward_pred_err = forward_scale * forward_loss(state2_hat_pred, \
                        state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    
    pred_action = inverse_model(state1_hat, state2_hat)

    inverse_pred_err = inverse_scale * inverse_loss(pred_action, \
                                        action.detach()).sum(dim=1).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err


