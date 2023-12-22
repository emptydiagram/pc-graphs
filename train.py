import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Fully connected PC graph
class FCPCGraph(nn.Module):
    def __init__(self, N, S, T=10, activation_fn='relu'):
        """Fully-connected Predictive coding graph (PC-graph) implementation.

        N -- total number of vertices (sensory + internal)
        S -- number of sensory vertices
        T -- number of time steps to run inference for"""
        super(FCPCGraph, self).__init__()
        self.N = N
        self.S = S
        self.x = nn.Parameter(torch.zeros(1,N), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(N,N), requires_grad=True)
        self.activation_fn = activation_fn

        if activation_fn == 'relu':
            self.f = nn.ReLU()
        elif activation_fn == 'sigmoid':
            self.f = nn.Sigmoid()
        elif activation_fn == 'tanh':
            self.f = nn.Tanh()
        else:
            raise Exception('Activation function not supported')

    def fprime(self, x):
        if self.activation_fn == 'relu':
            return (x > 0).float()
        elif self.activation_fn == 'sigmoid':
            return self.f(x) * (1 - self.f(x))
        elif self.activation_fn == 'tanh':
            return 1 - self.f(x)**2
        else:
            raise Exception('Activation function not supported')

    def inf_parameters(self):
        return [self.x]

    def wu_parameters(self):
        return [self.theta]

    def clamp_input(self, s):
        with torch.no_grad():
            self.x[0, :self.S] = s

    def inference_step(self):
        eps = self.x - self.f(self.x) @ self.theta
        x_grad = eps - self.fprime(self.x) * (eps @ self.theta.T)
        # set first S components to zero
        x_grad[0, :self.S] = 0.
        self.x.grad = x_grad

    def wu_step(self):
        eps = self.x - self.f(self.x) @ self.theta
        self.theta.grad = self.f(self.x).T @ eps


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_moving_collate_fn(device):
    def moving_collate_fn(batch):
        # batch is a list of pairs of length `batch_size`
        inputs, targets = zip(*batch)
        moved_inputs = torch.stack(inputs).to(device)
        moved_targets = torch.tensor(targets).to(device)
        return moved_inputs, moved_targets
    return moving_collate_fn


def train_mnist():
    num_epochs = 3
    batch_size = 1
    pcg_inf_steps = 10
    input_size = 28*28
    output_size = 10
    num_sensory_vertices = input_size + output_size
    num_vertices = 1500

    inf_lr = 0.5
    wu_lr = 0.00005

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device}')

    # load mnist
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    moving_collate_fn = make_moving_collate_fn(device=device)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate_fn)

    model = FCPCGraph(N=num_vertices, S=num_sensory_vertices, T=pcg_inf_steps, activation_fn='relu').to(device)
    model.to(device)

    inf_optimizer = torch.optim.SGD(model.inf_parameters(), lr=inf_lr)
    wu_optimizer = torch.optim.SGD(model.wu_parameters(), lr=wu_lr)

    print(type(model.parameters()))

    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        for batch_idx, (X, y) in enumerate(loader_train):
            X = X.view(batch_size, -1)
            Y = F.one_hot(y, num_classes=output_size).float()
            Z = torch.cat((X, Y), dim=1)
            if batch_idx % 500 == 0:
                print(f'Example {batch_idx}')
            model.clamp_input(Z)
            for t in range(pcg_inf_steps):
                inf_optimizer.zero_grad()
                model.inference_step()
                inf_optimizer.step()
            wu_optimizer.zero_grad()
            model.wu_step()
            wu_optimizer.step()



if __name__ == '__main__':
    set_random_seed(628318)
    train_mnist()