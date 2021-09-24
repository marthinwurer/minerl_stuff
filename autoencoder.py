import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
import torch.distributions as D

# Ignore warnings
import warnings

from torchvision.transforms import ToTensor, transforms
from tqdm import tqdm
from pytorch_msssim import ssim

from training_state import TrainingState, Hyperparameters
from utilities import conv2d_factory, unflatten, flat_shape, flatten

warnings.filterwarnings("ignore")


def spectral_loss(inputs, outputs, latents):
    in_spec = torch.view_as_real(torch.fft.rfft2(inputs, norm='ortho'))
    out_spec = torch.view_as_real(torch.fft.rfft2(outputs, norm='ortho'))
    return F.mse_loss(in_spec, out_spec)


def mse_loss(inputs, outputs, latents):
    return F.mse_loss(inputs, outputs)


def abs_loss(inputs, outputs, latents):
    return F.l1_loss(inputs, outputs)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def kld_loss(mu, log_var):
    loss = torch.mean(D.kl_divergence(D.Normal(mu, log_var),
                           D.Normal(torch.zeros_like(mu), torch.ones_like(log_var))))
    # loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return loss


class VAELatent(nn.Module):
    def __init__(self, latent_size, input_size):
        super().__init__()
        self.mu = nn.Linear(input_size, latent_size)
        self.logvar = nn.Linear(input_size, latent_size)

    def forward(self, input):
        x = input
        x = torch.flatten(x, 1)
        mu = self.mu(x)
        logvar = F.softplus(self.logvar(x))
        z = reparameterize(mu, logvar)
        return (z, mu, logvar)


def vae_loss(inputs, outputs, latents):
    z, mu, logvar = latents
    recon = F.mse_loss(inputs, outputs)
    # kl scaling from beta vae section 4.2
    kl_scale = z.shape[-1] / (64*64*3)
    kl = kld_loss(mu, logvar) * kl_scale
    return recon + kl, recon, kl


def vae_l1_loss(inputs, outputs, latents):
    z, mu, logvar = latents
    recon = F.l1_loss(inputs, outputs)
    # kl scaling from beta vae section 4.2
    kl_scale = z.shape[-1] / (64*64*3)
    kl = kld_loss(mu, logvar) * kl_scale
    return recon + kl, recon, kl


def vae_ssim_loss(inputs, outputs, latents):
    z, mu, logvar = latents
    recon = 1 - ssim(inputs, outputs, data_range=1, size_average=True, nonnegative_ssim=True)
    kl = kld_loss(mu, logvar)
    return recon + kl, recon, kl


def vae_mixed_ssim_loss(inputs, outputs, latents):
    z, mu, logvar = latents
    recon = 0.1 * (1 - ssim(inputs, outputs, data_range=1, size_average=True, nonnegative_ssim=True, win_size=5)) + F.l1_loss(inputs, outputs)
    kl = kld_loss(mu, logvar)
    return recon + kl, recon, kl


class ProGANAutoencoder(nn.Module):
    def __init__(self, latent_size, input_power, inner_power,
                 start_filters=16, max_filters=512, activation=nn.LeakyReLU):
        """

        Args:
            latent_size:
            input_power:
            inner_power: The power of 2 of the middle side length.
            start_filters:
            max_filters:
            activation:
        """
        super().__init__()
        self.latent_size = latent_size

        self.encoder = ProGANEncoder(input_power, inner_power, start_filters, max_filters, activation)

        self.fc_enc = nn.Linear(flat_shape(self.encoder.output_shape), latent_size)

        self.decoder = ProGANDecoder(latent_size, inner_power, input_power, start_filters, max_filters, activation)

    def forward(self, input):
        x = self.encoder(input)
        x = flatten(x)
        latent = self.fc_enc(x)
        output = self.decoder(latent)

        return output, latent

    def loss(self, input, output, latent):
        reconstruction = F.mse_loss(input, output)

        return reconstruction


class ProGANEncoder(nn.Module):
    """
    ProGAN discriminator layers go like this:

    image
    1x1 -> 16 (n=256)
    [
        3x3 -> 1x
        3x3 -> 2x
        downsample (n -> n//2)
    ]
    3x3 -> 1x (n=4)
    fully connected

    so what I really need this part to do is to do that inner loop, then tack
    on an extra conv layer or two at the end for that part. Also the start and
    end are really simple. I just go until the next size will be our 4x4

    """
    def __init__(self, input_power, output_power, start_filters=16, max_filters=512, activation=nn.LeakyReLU):
        """

        Args:
            input_power: the power of two that the side length of the input will be
            output_power: the power of two that the side length of the output will be
            activation: The activation function used by each layer
        """
        super().__init__()

        assert input_power > output_power

        layers = nn.ModuleList([])

        num_layers = input_power - output_power
        in_channels = start_filters
        out_channels = in_channels

        from_rgb = nn.Conv2d(3, start_filters, 1, 1)
        layers.append(from_rgb)
        layers.append(activation())

        for i in range(num_layers):
            out_channels = min(in_channels * 2, max_filters)
            layer = ProGANEncoderLayer(2, in_channels, out_channels, activation)
            in_channels = out_channels
            layers.append(layer)

        final = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        layers.append(final)

        self.layers = layers
        self.output_shape = (out_channels, 2**output_power, 2**output_power)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ProGANEncoderLayer(nn.Module):
    def __init__(self, num_layers: int, in_channels, out_channels, activation):
        super().__init__()
        num_starter = num_layers - 1
        if num_starter < 0:
            raise ValueError("Must have at least one layer")
        starter_layers = nn.ModuleList([])
        for i in range(num_starter):
            starter = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1)  # need padding to not shrink size
            starter_layers.append(starter)
            starter_layers.append(activation())
        grow = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        pool = nn.AvgPool2d(2, 2)
        self.starters = starter_layers
        self.layer = nn.Sequential(
            grow,
            activation(),
            pool
        )

    def forward(self, x):
        for layer in self.starters:
            x = layer(x)
        x = self.layer(x)
        return x


class ProGANDecoder(nn.Module):
    """
    latent (512)
    unflatten (512x1x1)
    deconv (1x1 -> 4x4)
    conv 3x3 -> x1
    {
        upsample x2
        conv 3x3 -> //2
        conv 3x3 -> x1
    }
    to_rgb 1x1 -> 3
    """
    def __init__(self, latent_size, input_power, output_power, end_filters=16, max_filters=512, activation=nn.LeakyReLU):
        """

        Args:
            input_power: the power of two that the side length of the input will be
            output_power: the power of two that the side length of the output will be
            activation: The activation function used by each layer
        """
        super().__init__()

        assert output_power > input_power

        reverse_layers = []

        num_layers = output_power - input_power
        in_channels = end_filters
        out_channels = in_channels

        to_rgb = nn.Conv2d(out_channels, 3, 1, 1)

        for i in range(num_layers):
            in_channels = min(out_channels * 2, max_filters)
            layer = ProGANDecoderLayer(2, in_channels, out_channels, activation)
            out_channels = in_channels
            reverse_layers.append(layer)

        ordered_layers = reversed(reverse_layers)

        unflatten = nn.ConvTranspose2d(latent_size, out_channels, kernel_size=2**input_power)

        top_same = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        top_block = nn.Sequential(
            unflatten,
            activation(),
            top_same,
            activation()
        )

        self.layers = nn.ModuleList([])
        self.layers.append(top_block)
        self.layers.extend(ordered_layers)
        self.layers.append(to_rgb)
        self.output_shape = (end_filters, 2**output_power, 2**output_power)

    def forward(self, x):
        # need to unflatten the input
        x = unflatten(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ProGANDecoderLayer(nn.Module):
    def __init__(self, num_layers: int, in_channels, out_channels, activation):
        super().__init__()
        num_same = num_layers - 1
        if num_same < 0:
            raise ValueError("Must have at least one layer")

        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        shrink = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)

        self.layer = nn.Sequential(
            upsample,
            shrink,
            activation(),
        )

        same_layers = nn.ModuleList([])
        for i in range(num_same):
            starter = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)  # need padding to not shrink size
            same_layers.append(starter)
            same_layers.append(activation())

        self.same = same_layers

    def forward(self, x):
        x = self.layer(x)
        for layer in self.same:
            x = layer(x)
        return x


def train_autoencoder(net, optimizer, device, trainset, trainloader, batch_size, epochs, callback=None):
    train_batches = math.ceil(len(trainset) / batch_size)
    running_loss = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        # loss_steps = 8000 / batch_size
        loss_steps = 5

        with tqdm(enumerate(trainloader, 0), total=train_batches, unit="batch") as t:
            for i, data in t:
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, latents = net(inputs)
                loss = net.loss(inputs, outputs, latents)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % loss_steps == loss_steps - 1:  # print every N mini-batches
                    string = '[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / loss_steps)
                    t.set_postfix_str(string)
                    running_loss = 0.0
            if callback:
                callback()


def train_ae_with_state(training_state: TrainingState, device, trainset, trainloader, epochs):
    for i in range(epochs):
        train_autoencoder(training_state.model, training_state.optimizer, device, trainset, trainloader, training_state.hyper.batch_size, 1, callback=None)
        training_state.training_steps += 1

        # save checkpoint
        filename = "saved_nets/autoencoder_cp_%s.tar" % training_state.training_steps
        training_state.save_state(filename)


def main():
    # plt.ion()  # interactive mode

    cont = True

    filename = "saved_nets/autoencoder/autoencoder_state.tar"

    if os.path.exists(filename) and not cont:
        raise FileExistsError(filename)

    # BATCH_SIZE = 128
    BATCH_SIZE = 32
    # BATCH_SIZE = 8
    LEARNING_RATE = 0.00001
    EPOCHS = 30
    MOMENTUM = 0.9
    IN_POWER = 8

    in_dim = 2 ** IN_POWER

    hyper = Hyperparameters(BATCH_SIZE, LEARNING_RATE)

    net_transform = transforms.Compose([
        transforms.Resize(in_dim),
        transforms.RandomCrop(in_dim, pad_if_needed=True),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = VGImagesDataset(root_dir=VG_PATH, transform=net_transform)

    net = ProGANAutoencoder(512, IN_POWER, 2)
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    train_state = TrainingState(net, optimizer, hyper)

    if cont:
        train_state.load_state(filename)
    else:
        # test saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        train_state.save_state(filename)

    train_ae_with_state(train_state, device, dataset, trainloader, EPOCHS)

    train_state.save_state(filename)

    path = "saved_nets/autoencoder.mod"
    print("Saving Model to %s" % path)
    torch.save(net.state_dict(), path)




if __name__ == "__main__":
    main()


