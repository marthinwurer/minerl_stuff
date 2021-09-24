
import matplotlib.pyplot as plt
import numpy as np
import torch
import minerl
from tqdm import tqdm

from torch import nn, optim
import torch.nn.functional as F
from minerl.data import BufferedBatchIter

from networks import WMAutoencoder

# BATCH_SIZE = 512
# BATCH_SIZE = 256
# BATCH_SIZE = 128
BATCH_SIZE = 32
# BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 30
MOMENTUM = 0.9
# IN_POWER = 8
# IN_POWER = 6
in_dim = 64


model = WMAutoencoder().cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-4)

data = minerl.data.make(
    'MineRLObtainDiamond-v0', "/home/marthinwurer/projects/acx_minerl/data")
bbi = BufferedBatchIter(data)


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


    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss


running_loss = 0
loss_steps = 5
epoch = 0

iterator = bbi.buffered_batch_iter(batch_size=BATCH_SIZE, num_epochs=1)
with tqdm(enumerate(iterator, 0), unit="batch") as t:
    for i, data in t:
        # get the inputs
        current_state, action, reward, next_state, done = data
        image = torch.from_numpy(current_state['pov'].transpose(0, 3, 1, 2)).cuda() / 255

        loss = train_batch(image, model, optimizer)

        if loss is None or torch.isnan(loss).any():
            print("There's a NaN loss!")
            from IPython.core.debugger import Pdb;

            Pdb().set_trace()
            break

        # print statistics
        running_loss += loss.item()
        if i % loss_steps == loss_steps - 1:  # print every N mini-batches
            string = '[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / loss_steps)
            t.set_postfix_str(string)
            running_loss = 0.0

