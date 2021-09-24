from coordconv import *
from spectralpool import SpectralPool2d
from utilities import flatten, unflatten, nan_canary


class AdvancedAutoencoder(nn.Module):
    def __init__(self, latent_size=128, activation=nn.LeakyReLU):
        super().__init__()

        self.encoder = AAEEncoder(activation)
        self.latent = AAELatent(latent_size, activation)
        self.decoder = AAEDecoder(latent_size, activation)

    def forward(self, input):
        x = input
        x = self.encoder(x)
        latent = self.latent(x)
        output = self.decoder(latent)
        return output, latent


class AAEEncoder(nn.Module):
    def __init__(self, activation=nn.LeakyReLU):
        super().__init__()

        self.encoder = nn.Sequential(
            CoordConv2d(3, 32, 3, padding=1),  # 64
            activation(),
            SpectralPool2d(.5),  # 32
            CoordConv2d(32, 64, 3, padding=1),
            activation(),
            SpectralPool2d(.5),  # 16
            CoordConv2d(64, 128, 3, padding=1),
            activation(),
            SpectralPool2d(.5),  # 8
            CoordConv2d(128, 256, 3, padding=1),
            activation(),
            SpectralPool2d(.25),  # 2
            CoordConv2d(256, 256, 3, padding=1),
            activation(),
        )

    def forward(self, input):
        x = input
        x = self.encoder(x)
        return x


class AAEDecoder(nn.Module):
    def __init__(self, latent_size, activation=nn.LeakyReLU):
        super().__init__()

        self.decompress = nn.Linear(latent_size, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4),  # 4x4
            activation(),
            SpectralPool2d(2),  # 8x8
            CoordConv2d(256, 128, 3, padding=1),
            activation(),
            SpectralPool2d(2),  # 16
            CoordConv2d(128, 64, 3, padding=1),
            activation(),
            SpectralPool2d(2),  # 32
            CoordConv2d(64, 32, 3, padding=1),
            activation(),
            SpectralPool2d(2),  # 64
            CoordConv2d(32, 3, 3, padding=1),
        )

    def forward(self, input):
        x = input
        x = self.decompress(x)
        x = unflatten(x)
        x = self.decoder(x)
        return x


class AAELatent(nn.Module):
    def __init__(self, latent_size, activation=nn.LeakyReLU):
        super().__init__()
        self.compress = nn.Linear(1024, latent_size)

    def forward(self, input):
        x = input
        x = flatten(x)
        x = self.compress(x)
        return x

