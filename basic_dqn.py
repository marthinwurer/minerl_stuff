
import numpy as np
import torch
# import minerl
from tqdm import tqdm
import random

from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities import flatten, unflatten, to_batch_shape, to_torch_channels
from collections import defaultdict
import coordconv

import collections

from ZerO import init_ZerO
from networks import SkippableLayerNorm
from lion_pytorch import Lion

# replay buffer. Store (s, a, r, s_n, d) tuples
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = []
        self.max_size = max_size
        self.insert_at = 0
    
    def add(self, s, a, r, s_n, d):
        if len(self.buffer) < self.max_size:
            self.buffer.append((s, a, r, s_n, d))
        else:
            self.buffer[self.insert_at] = (s, a, r, s_n, d)
            self.insert_at = (self.insert_at + 1) % self.max_size
    
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def add_all(self, sarsd):
        for i in sarsd:
            buffer.add(*i)
            
            
def sample_training_batch(buffer, batch_size):
    batch = buffer.sample_batch(batch_size)
    s = np.zeros((batch_size, 64, 64, 3), dtype='float32')
    a = np.zeros((batch_size, 1), dtype='int64')
    r = np.zeros((batch_size, 1), dtype='float32')
    s_n = np.zeros((batch_size, 64, 64, 3), dtype='float32')
    d = np.zeros((batch_size, 1), dtype='float32')

    
    for i, x in enumerate(batch):
        s[i] = x[0]
        a[i] = x[1]
        r[i] = x[2]
        s_n[i] = x[3]
        d[i] = x[4]
    
    s = torch.from_numpy(s).permute((0, 3, 1, 2)).cuda() / 255
    s_n = torch.from_numpy(s_n).permute((0, 3, 1, 2)).cuda() / 255
    a = torch.from_numpy(a).cuda()
    r = torch.from_numpy(r).cuda()
    d = torch.from_numpy(d).cuda()
    
    return s, a, r, s_n, d


class EnvSamplingWrapper:
    def __init__(self, env):
        self.env = env
        self.obs = env.reset()
        self.step_count = 0
        self.total_reward = 0
    
    def sample(self, count, policy):
        tuples = []
        rewards = []
        for i in range(count):
            self.step_count += 1
            action = policy(self.obs)
            observation, reward, done, info = env.step(action)
            tuples.append((self.obs, action, reward, observation, done))
            self.obs = observation
            self.total_reward += reward
            if done:
                self.obs = env.reset()
                rewards.append((self.total_reward, self.step_count))
                self.total_reward = 0
        
        return tuples, rewards


def prod(x):
    total = 1
    for i in x:
        total *= i
    return total


class DQN_Net(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
#         print(input_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, action_size),
        )
        with torch.no_grad():
            self.net[-1].weight[:,:] = 0
    
    def forward(self, x):
#         print(x.shape)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.net(x)
        return x

    
def dqn_policy(model, obs):
    # scale the obs, change to model shape
    image = torch.from_numpy(to_batch_shape(obs)).permute((0, 3, 1, 2)).cuda() / 255
    with torch.no_grad():
        outputs = model(image)
        action = torch.argmax(outputs).item()
    
    return action


def train_batch(model, target, buffer, optimizer):
    # get the inputs
    
    # sample from the buffer and preprocess next qs
    s, a, r, s_n, d = sample_training_batch(buffer, 32)
#     print(d)
    
    # get the next qs
    with torch.no_grad():
        outputs = target(s_n)
        q_n, a_n = torch.max(outputs, 1)
        # Avoid potential broadcast issue
        q_n = q_n.view(-1, 1)
#         print(q_n.shape)
        targets = r * (1.0) + q_n * (1 - d) * 0.99

    # zero the parameter gradients
    optimizer.zero_grad(set_to_none=True)
    
    # forward + backward + optimize
    outputs = model(s)
#     print(a.shape)
    
    # get the target action to diff the target with
    v = torch.gather(outputs, 1, a)
#     print(v.shape, outputs.shape)

    if torch.isnan(outputs).any():
        print("There's a NaN output!")
        return None
    loss = F.smooth_l1_loss(v, targets)

    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.detach().item()


class DQN_Policy:
    def __init__(self, model, epsilon, sampler):
        self.model = model
        self.epsilon = epsilon
        self.sampler = sampler
    
    def __call__(self, obs):
        if random.random() < self.epsilon:
            return self.sampler()
        return dqn_policy(self.model, obs)

    
# from https://github.com/ghliu/pytorch-ddpg/blob/e9db328ca70ef9daf7ab3d4b44975076ceddf088/util.py#L26
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

        
def dqn_loop(model, target, env_wrapper, buffer, optimizer, steps, STATS):
    running_loss = None
    
    next_exploration = 0.95
    with tqdm(range(steps), unit="batch") as t:
        for i in t:
            
            if i % 1000 == 0:
                policy = DQN_Policy(model, next_exploration, env_wrapper.env.action_space.sample)
                next_exploration -= 0.05
                next_exploration = max(next_exploration, 0.05)
            
            sarsd, rew = env_wrapper.sample(1, policy)
            buffer.add_all(sarsd)
            STATS['returns'].extend(rew)
            
            if (i + 1) % 4 != 0:
                continue

            loss = train_batch(model, target, buffer, optimizer)
            STATS["loss"].append(loss)

            if i % 10000 == 0:
                soft_update(target, model, 1.0)
#             soft_update(target, model, 0.000001)
            if running_loss is None:
                running_loss = loss
            running_loss = running_loss * 0.999 + loss * 0.001
            if (i + 1) % 100 == 0:  # print every N mini-batches
                string = 'loss: %.8f' % (
                    running_loss
                )
                t.set_postfix_str(string)