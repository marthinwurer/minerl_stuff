import os

import torch


class TrainingState(object):

    def __init__(self, model, optimizer, hyper=None, training_steps=0):
        self.hyper = hyper
        self.training_steps = training_steps

        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        return {
            "hyperparameters": self.hyper,
            "training_steps": self.training_steps,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def save_state(self, filename):
        torch.save(self.state_dict(), filename)

    def load_state(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.hyper = checkpoint['hyperparameters']
            self.training_steps = checkpoint['training_steps']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (steps: {})"
                  .format(filename, self.training_steps))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            raise FileNotFoundError()


class Hyperparameters(object):

    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate







