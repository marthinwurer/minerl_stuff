import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv
from tqdm.auto import tqdm

from coordconv import CoordConv2d, AddCoords
from utilities import nan_canary, flatten, unflatten


def generate_image(coord, shape):
    image = np.zeros(shape)
    image[coord[0], coord[1]] = 1
    return image


def generate_single_dot_dataset(height, width):
    coordinates = []
    for h in range(height):
        for w in range(width):
            coordinates.append((h, w))

    images = []
    for coord in coordinates:
        images.append((generate_image(coord, (height, width)), coord))

    return images


def random_split(data, train_split):
    """
    Randomly split the data into a train and test split
    """
    to_shuffle = data.copy
    random.shuffle(to_shuffle)
    train_end = int(len(to_shuffle) * train_split)
    train = to_shuffle[:train_end]
    test = to_shuffle[train_end:]
    return train, test


def find_quadrant(coord, shape):
    center_y = shape[0] // 2
    center_x = shape[0] // 2
    y = coord[0] - center_y
    x = coord[1] - center_x
    if y < 0:
        if x > 0:
            return 1
        else:
            return 2
    else:
        if x > 0:
            return 3
        else:
            return 4


def quadrant_split(data, quadrant, shape):
    """
    Split the data into a train and test split by quadrant
    """
    train = []
    test = []
    for item in data:
        image, coord = item
        if quadrant == find_quadrant(coord, shape):
            test.append(item)
        else:
            train.append(item)

    return train, test


class CoordConvTranspose2d(conv.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.ConvTranspose2d(in_channels + self.rank + int(with_r), out_channels,
                                       kernel_size, stride, padding, output_padding=0,
                                       dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, output_size=None):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input)
        nan_canary(out)
        out = self.conv(out)

        return out


class WMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CoordConv2d(3, 32, 4, stride=2)
        self.conv2 = CoordConv2d(32, 64, 4, stride=2)
        self.conv3 = CoordConv2d(64, 128, 4, stride=2)
        self.conv4 = CoordConv2d(128, 256, 4, stride=2)  # 2x2x256

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.layer_norm(x, x.size()[1:])
#         print(x.shape)
        nan_canary(x)
        x = F.relu(self.conv2(x))
#         print(x.shape)
        nan_canary(x)
        x = F.layer_norm(x, x.size()[1:])
#         print(x.shape)
        nan_canary(x)
        x = F.relu(self.conv3(x))
#         print(x.shape)
        x = F.layer_norm(x, x.size()[1:])
#         print(x.shape)
        nan_canary(x)
        x = F.relu(self.conv4(x))
#         print(x.shape)
        x = F.layer_norm(x, x.size()[1:])
        nan_canary(x)
#         print(x.shape)
        return x


class WMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
#         self.deconv1 = CoordConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = CoordConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = CoordConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = CoordConvTranspose2d(32, 16, 6, stride=2)
        self.to_rgb = CoordConv2d(16, 3, 1, stride=1)


    def forward(self, x):
        nan_canary(x)
#         print(x.shape)
        x = F.layer_norm(x, x.size()[1:])
#         print(x.shape)
        nan_canary(x)
        x = self.deconv1(x)
        nan_canary(x)
        x = F.relu(x)
#         print(x.shape)
        nan_canary(x)
        x = F.relu(self.deconv2(F.layer_norm(x, x.size()[1:])))
#         print(x.shape)
        nan_canary(x)
        x = F.relu(self.deconv3(F.layer_norm(x, x.size()[1:])))
#         print(x.shape)
        nan_canary(x)
        x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.deconv4(x))  # activation and norm removed so it's a fully linear layer
        x = self.to_rgb(x)
        return x

class WMLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(1024, 128)
        self.decompress = nn.Linear(128, 1024)

    def forward(self, x):
        x = flatten(x)
        x = self.compress(x)
        nan_canary(x)
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


def train_batch(inputs, model, optimizer):
    # get the inputs
    inputs = inputs.cuda()

    if torch.isnan(inputs).any():
        print("There's a NaN input!")
        return None

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, latents = model(inputs)

    if torch.isnan(outputs).any():
        print("There's a NaN output!")
        return None
    loss = model.loss(inputs, outputs, latents)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(trainloader, train_batches, model, optimizer):
    running_loss = 0.0
    epoch = 0.0
    loss_steps = 5
    with tqdm(enumerate(trainloader, 0), total=train_batches, unit="batch") as t:
        for i, data in t:
            # get the inputs
            loss = train_batch(data, model, optimizer)


            if loss is None or torch.isnan(loss).any():
                print("There's a NaN loss!")
                break

            # print statistics
            running_loss += loss.item()
            if i % loss_steps == loss_steps - 1:  # print every N mini-batches
                string = '[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / loss_steps)
                t.set_postfix_str(string)
                running_loss = 0.0