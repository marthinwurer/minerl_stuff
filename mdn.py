import torch
import torch.distributions as D

from torch import nn


class MDN(nn.Module):
    def __init__(self, input_size, output_size, num_gaussians):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.num_gaussians = num_gaussians
        num_outs = output_size * num_gaussians
        self.pi = nn.Sequential(nn.Linear(input_size, num_outs), nn.Softplus())
        self.mu = nn.Linear(input_size, num_outs)
        self.sigma = nn.Sequential(nn.Linear(input_size, num_outs), nn.Softplus())

    def forward(self, input):
        pi = self.pi(input).view(input.shape[0], self.output_size, self.num_gaussians)
        mu = self.mu(input).view(input.shape[0], self.output_size, self.num_gaussians)
        sigma = self.sigma(input).view(input.shape[0], self.output_size, self.num_gaussians)
        return pi, sigma, mu


def distribution(pi, sigma, mu):
    mix = D.Categorical(pi)
    comp = D.Normal(mu, sigma)

    gm = D.MixtureSameFamily(mix, comp)
    return gm


def gnll_loss(pi, sigma, mu, target):
    gm = distribution(pi, sigma, mu)
    log_likelihood = gm.log_prob(target)
    return -torch.mean(log_likelihood)
