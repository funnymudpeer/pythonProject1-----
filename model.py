# 前向卷积模型
import torch.nn.functional
from torch.nn import Module
from torch import nn
import scipy.io as scio


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        G_data = scio.loadmat('./G.mat')
        G = torch.tensor(G_data['G'])
        height = G.shape[0]
        width = G.shape[1]
        G = torch.reshape(G, (1, height, width))
        zero = torch.zeros((1, height, width))
        self.G = torch.zeros((3, 3, height, width))
        self.G[0, :, :, :] = torch.cat([G, zero, zero], dim=0)
        self.G[1, :, :, :] = torch.cat([zero, G, zero], dim=0)
        self.G[2, :, :, :] = torch.cat([zero, zero, G], dim=0)
        self.ori = nn.Parameter(torch.zeros((1, 3, 480, 720)), requires_grad=True)

    def forward(self):
        # print(self.my_ori.data.shape, self.G.shape)
        y = nn.functional.conv2d(self.ori, self.G, padding='same')
        return y