from torch import nn, optim

from autoencoder import VAELatent
from coordconv import *
from utilities import flatten, unflatten, nan_canary

import torch.nn.functional as F


class SkippableLayerNorm(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm
        
    def forward(self, x):
        if self.norm:
            x = F.layer_norm(x, x.size()[1:])
        return x


class SimpleLayerNorm(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        
    def forward(self, x):
        x = F.layer_norm(x, x.size()[1:])
        return x


class WMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CoordConv2d(3, 32, 4, stride=2)
        self.conv2 = CoordConv2d(32, 64, 4, stride=2)
        self.conv3 = CoordConv2d(64, 128, 4, stride=2)
        self.conv4 = CoordConv2d(128, 256, 4, stride=2)  # 2x2x256
        self.activation = F.silu

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[1:])
        x = self.activation(self.conv2(x))
        x = F.layer_norm(x, x.size()[1:])
        x = self.activation(self.conv3(x))
        x = F.layer_norm(x, x.size()[1:])
        x = self.activation(self.conv4(x))
        x = F.layer_norm(x, x.size()[1:])
        return x


class WMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.deconv1 = CoordConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = CoordConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = CoordConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = CoordConvTranspose2d(32, 16, 6, stride=2)
        self.to_rgb = CoordConv2d(16, 3, 1, stride=1)
        self.activation = F.silu

    def forward(self, input):
        x = input
        x = F.layer_norm(x, x.size()[1:])
        x = self.deconv1(x)
        x = self.activation(x)
        x = self.activation(self.deconv2(F.layer_norm(x, x.size()[1:])))
        x = self.activation(self.deconv3(F.layer_norm(x, x.size()[1:])))
        x = F.layer_norm(x, x.size()[1:])
        x = F.sigmoid(self.deconv4(x))  # activation and norm removed so it's a fully linear layer
        x = self.to_rgb(x)
        return x


class WMLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(1024, 256)
        self.decompress = nn.Linear(256, 1024)

    def forward(self, x):
        x = flatten(x)
        x = self.compress(x)
        latent = x
        x = self.decompress(x)
        x = unflatten(x)
        return x, latent


class WMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WMEncoder()
        self.latent = WMLatent()
        self.decoder = WMDecoder()

    def forward(self, x):
        x = self.encoder(x)
        #         print(x.shape)
        x, latent = self.latent(x)
        x = self.decoder(x)
        return x, latent

    def loss(self, inputs, outputs, latents):
        return F.mse_loss(inputs, outputs)


class WM_VAE(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()
        activation = nn.ReLU
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            activation(),
            nn.Conv2d(32, 64, 4, stride=2),
            activation(),
            nn.Conv2d(64, 128, 4, stride=2),
            activation(),
            nn.Conv2d(128, 256, 4, stride=2),
            activation(),
        )

        self.latent = VAELatent(latent_size, 1024)

        self.unlatent = nn.Linear(latent_size, 1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            activation(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        latent = self.latent(encoded)
        x = self.unlatent(latent[0])
        x = unflatten(x)
        decoded = self.decoder(x)
        return decoded, latent


"""
Inputs:
    xp: observation (pov)
    xv: extra latent (vector)
    a: action

Outputs:
    rh: reward
    xph: reconstructed pov
    xvh: reconstructed vector
    h: hidden state
    z: state representation
    zh: stochastic state representation
    vh: critic
    ah: actor

Networks:
    encoder: xp -> xp_latent
    decoder: (h, z) -> xph, xvh
    recurrent: (h, z, a) -> h
    representation: (h, xp, xv) -> z
    rep_distribution h -> zh
    reward: (h, z) -> rh
    actor: z -> ah
    critic: z -> v

"""


class VisionEncoder(nn.Module):
    def __init__(self, latent_size=128, activation=nn.ELU, embed=True, norm=False, coord=False):
        super().__init__()
        conv = nn.Conv2d
        if coord:
            conv = CoordConv2d
        
        layers = [
            conv(3, 32, 4, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(32, 64, 4, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(64, 128, 4, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(128, 256, 4, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            nn.Flatten(),
        ]
        if embed:
            layers.append(nn.Linear(1024, latent_size))
#             layers.append(SkippableLayerNorm(norm))
            layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*layers)

    def forward(self, input):
        x = self.encoder(input)
        return x


class VisionDecoder(nn.Module):
    def __init__(self, latent_size=128, activation=nn.ELU, norm=False, coord=False):
        super().__init__()
        conv = nn.ConvTranspose2d
        if coord:
            conv = CoordConvTranspose2d

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.Unflatten(-1, (1024, 1, 1)),
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(128, 64, 5, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(64, 32, 6, stride=2),
            activation(),
            SkippableLayerNorm(norm),
            conv(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.decoder(input)
        return x


class CCDecoder(nn.Module):
    def __init__(self, latent_size=128, activation=nn.SiLU, norm=True):
        super().__init__()
        
        self.latent_size = latent_size
        
        self.decoder = nn.Sequential(
            CoordConv2d(latent_size, 128, 1),
            activation(),
            SkippableLayerNorm(norm),
            CoordConv2d(128, 64, 1),
            activation(),
            SkippableLayerNorm(norm),
            CoordConv2d(64, 32, 1),
            activation(),
            SkippableLayerNorm(norm),
            CoordConv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = torch.tile(input.view(-1, self.latent_size, 1, 1), (64, 64))
        x = self.decoder(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, latent_size=128, activation=nn.SiLU, norm=True, coord=True):
        super().__init__()
        self.norm = norm
        self.encoder = VisionEncoder(latent_size, activation, coord=coord, norm=norm)
        self.decoder = VisionDecoder(latent_size, activation, coord=coord, norm=norm)
#         self.decoder = CCDecoder(latent_size, activation, norm=norm)
    
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded, encoded


class VisionVAE(nn.Module):
    def __init__(self, latent_size=128, activation=nn.SiLU, norm=True, coord=True):
        super().__init__()
        self.norm = norm
        self.encoder = VisionEncoder(latent_size, activation, embed=False, coord=coord, norm=norm)
        self.latent = VAELatent(latent_size, 1024)
        self.decoder = VisionDecoder(latent_size, activation, coord=coord, norm=norm)

    #         self.decoder = CCDecoder(latent_size, activation, norm=norm)

    def forward(self, input):
        encoded = self.encoder(input)
        vae_out = self.latent(encoded)
        decoded = self.decoder(vae_out[0])
        return decoded, vae_out


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, padding=3, stride=1, activation=nn.GELU, norm=SimpleLayerNorm):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=padding, stride=stride),
            norm(),
            nn.Conv2d(channels, channels * 4, 1),
            activation(),
            nn.Conv2d(channels * 4, channels, 1),
        )
    
    def forward(self, x):
        x = x + self.layer(x)
        return x




