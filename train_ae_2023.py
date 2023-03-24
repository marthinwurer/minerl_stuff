"""
In this file, we train the autoencoder for world models.
"""
import numpy as np
import torch
import minerl
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_preprocessing import MineRlSequenceDataset, MineRlImageDataset
from ZerO import init_ZerO
from networks import SkippableLayerNorm
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from utilities import flatten, unflatten, to_batch_shape, to_torch_channels
from collections import defaultdict


BATCH_SIZE = 256
LEARNING_RATE = 0.0001


class AELatent(nn.Module):
    def __init__(self, latent_size, input_size, norm=nn.LazyBatchNorm1d):
        super().__init__()
        self.mu = nn.Linear(input_size, latent_size)
    
    def forward(self, input):
        x = input
        x = torch.flatten(x, 1)
        mu = self.mu(x)
        return mu


class WM_AE(nn.Module):
    def __init__(self,
                 latent_size=128,
                 activation=nn.ReLU,
                 conv=nn.Conv2d,
                 norm=nn.LazyBatchNorm2d,
                 deconv=nn.ConvTranspose2d,
                ):
        super().__init__()

        self.encoder = nn.Sequential(
            conv(3, 32, 4, stride=2),
            activation(),
            norm(),
            conv(32, 64, 4, stride=2),
            activation(),
            norm(),
            conv(64, 128, 4, stride=2),
            activation(),
            norm(),
            conv(128, 256, 4, stride=2),
            activation(),
            norm(),
        )
        
        self.latent = AELatent(256, 1024, norm=norm)
        self.unlatent = nn.Linear(256, 1024)
        
        self.decoder = nn.Sequential(
            norm(),
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            activation(),
            norm(),
            deconv(128, 64, 5, stride=2),
            activation(),
            norm(),
            deconv(64, 32, 6, stride=2),
            activation(),
            norm(),
            deconv(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )
    
    def forward(self, input):
        encoded = self.encoder(input)
        latent = self.latent(encoded)
        x = latent
        if isinstance(latent, tuple):
            x = latent[0]
            
        x = self.unlatent(x)
        x = unflatten(x)
        decoded = self.decoder(x)
        return decoded, latent


def train_batch(inputs, model, optimizer, loss_func):
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
    loss = loss_func(inputs, outputs, latents)
    if isinstance(loss, tuple):
        loss[0].backward()
    else:
        loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss


def rae_loss(inputs, outputs, latents):
    recon = F.mse_loss(inputs, outputs)
    l2 = torch.linalg.vector_norm(latents) * 1e-6
    return recon + l2, recon, l2


def lr_schedule(step):
    """
    Schedule the learning rate each training step.
    
    Start with a warmup over one epoch, then stay constant
    
    output is multiplied by the initial learning rate to find the final lr
    """
    EPOCH_SIZE = 500
    e = 0.00000001
    if step < EPOCH_SIZE:
        # linear ramp from e to LR
        return 1 - ((1 - e) * ((EPOCH_SIZE - step) / EPOCH_SIZE))
    return 1


def main():
    dataset = MineRlImageDataset("data/npy_obtain_diamond_all")
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = WM_AE(activation=nn.SiLU, norm=SkippableLayerNorm)
    model.apply(init_ZerO)
    with torch.no_grad():
        model.latent.mu.weight[:,:] = 0
    model.cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00025)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_func = rae_loss
    STATS = defaultdict(list)
    
    loss_steps = 50
    epochs = 10
    model.train()

    for epoch in range(epochs):
        with tqdm(enumerate(train_dataloader, 0), unit="batch") as t:
            running_loss = 0
            for i, data in t:
                # get the inputs
                image = data.transpose(-1, 1).cuda() / 255

                full_loss = train_batch(image, model, optimizer, loss_func)

                if isinstance(full_loss, tuple):
                    loss = full_loss[0].item()
                    STATS['raw_losses'].append(loss)
                    STATS['recons'].append(full_loss[1].item())
                    STATS['kls'].append(full_loss[2].item())
                else:
                    loss = full_loss.item()
                    STATS['raw_losses'].append(loss)

                lr_scheduler.step()
                if hasattr(loss_func, "step"):
                    loss_func.step()

                # print statistics
                running_loss += loss
                if i % loss_steps == loss_steps - 1:  # print every N mini-batches
                    string = '[%d, %5d] loss: %.8f lr=%s' % (
                        epoch + 1, i + 1, running_loss / loss_steps, lr_scheduler.get_last_lr()
                    )
                    t.set_postfix_str(string)
                    STATS['losses'].append(running_loss / loss_steps)
                    running_loss = 0.0
        
        # at the end of each epoch, save a snapshot
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H_%M_%S")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"./models/ae_checkpoint_{now_str}-{epoch}.mdl")

if __name__ == "__main__":
    main()
