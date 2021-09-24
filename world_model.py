#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.rcParams['figure.figsize'] = [15, 10]


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import minerl
from tqdm import tqdm

from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities import flatten, unflatten, to_batch_shape, to_torch_channels

import autoencoder
from AdvancedAutoencoder import AdvancedAutoencoder
from networks import WMAutoencoder, WM_VAE, VisionEncoder, VisionDecoder, VAELatent
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset_preprocessing import MineRlSequenceDataset
import mdn


# In[22]:


# BATCH_SIZE = 512
# BATCH_SIZE = 256
# BATCH_SIZE = 128
# BATCH_SIZE = 64
# BATCH_SIZE = 32
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 30
MOMENTUM = 0.9
# IN_POWER = 8
# IN_POWER = 6
in_dim = 64


# In[23]:


dataset = MineRlSequenceDataset("/home/marthinwurer/projects/acx_minerl/data/npy_obtain_diamond", 32)


# In[24]:


train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


# In[25]:


train_features = next(iter(train_dataloader))
train_features[0].shape


# In[29]:


plt.imshow(train_features[0][2][3].permute(1, 2, 0))


# In[80]:


class WorldModel(nn.Module):
    def __init__(self, hidden_size=600, vision_size=256, vec_size=64, action_size=64):
        super().__init__()
        obs_size = vision_size + vec_size
        self.encoder = VisionEncoder(vision_size, embed=False)
        self.decoder = VisionDecoder(vision_size)
        self.latent = VAELatent(vision_size, 1024)
        self.rnn = nn.GRU(obs_size + action_size, hidden_size)
        self.transition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            mdn.MDN(hidden_size, obs_size+2, 5),  # extra 2 are for reward and done
        )
    
    def forward(self, povs, vecs, actions, hidden=None):
        timesteps, batch_size, C, H, W = povs.shape
        for_cnn = povs.view(timesteps * batch_size, C, H, W)
        encoded = self.encoder(for_cnn)
        z, mu, logvar = self.latent(encoded)
        recons = self.decoder(z).view(timesteps, batch_size, C, H, W)
        pov_obs = mu.view(timesteps, batch_size, -1)
        
        obs = torch.cat((pov_obs, vecs), -1)
        
        h, final_h = self.rnn(torch.cat((obs, actions), -1), hidden)

        trans_out = self.transition(h.view(timesteps * batch_size, -1))

        return (z, mu, logvar), recons, obs, h, trans_out


# In[81]:


model = WorldModel().cuda()


# In[82]:


data = [x.cuda() for x in train_features]
povs, vecs, actions, rewards, dones = data


# In[83]:


outs = model(povs, vecs, actions)


# In[84]:


vae_out, recons, obs, h, trans_out = outs


def wm_loss(outs, povs, vecs, actions, rewards, dones):
    vae_out, recons, obs, h, trans_out = outs
    recons_loss = autoencoder.vae_loss(povs, recons, vae_out)[0]
    
    # build the target for prediction loss
    #todo shift the obs or figure out a better way to do this
    pred_target = torch.cat((obs, torch.log2(rewards + 1), dones), -1).view(obs.shape[0]*obs.shape[1], -1)
    pred_loss = mdn.mdn_loss(*trans_out, pred_target)
    
    print(recons_loss, pred_loss)
    
    return recons_loss + pred_loss
    


# In[96]:


loss = wm_loss(outs, povs, vecs, actions, rewards, dones)
