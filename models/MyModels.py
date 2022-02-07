import torch
from torch import nn
import math
import numpy as np
import pdb

class HistByNorm(nn.Module):
    def __init__(self, centers, widths):
        super(HistByNorm, self).__init__()
        self.hist_centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=True)
        self.hist_widths = nn.Parameter(torch.tensor(widths, dtype=torch.float32), requires_grad=True)

        
    def forward(self, x): #[72,2048,16,8]
#         dist = Normal(self.hist_centers, self.hist_widths)
#         pdb.set_trace()
#         x = torch.exp(dist.log_prob(x))
        res = torch.zeros((x.shape[0] * x.shape[1], len(self.hist_centers))).cuda()
        for i in range(len(self.hist_centers)):
            inputt = x.view(x.shape[0] * x.shape[1], -1)
            res[:, i] = torch.mean(self.norm(inputt, self.hist_centers[i], self.hist_widths[i]), 1)
        return res.view(x.shape[0], -1)
    
    def norm(self, x, mu, sigma):
        return (1 / (sigma * torch.sqrt(torch.tensor(2 * math.pi)))) * torch.exp((-0.5*((x - mu)/sigma)**2))

    

    
class HistByProf(nn.Module):
    def __init__(self, edges):
        super(HistByProf, self).__init__()
        self.hist_edges = edges
        self.norm_centers = []
        self.sigma = 0.39
        self.nbins = len(edges) + 1
        
        for i in range(len(edges)-1):
            self.norm_centers.append((self.hist_edges[i] + self.hist_edges[i+1])/2)
            
        self.norm_centers = nn.Parameter(torch.tensor(self.norm_centers, dtype=torch.float32), requires_grad=True)
        self.sigmoid_semi_centers = nn.Parameter(torch.tensor([self.hist_edges[0], self.hist_edges[-1]], dtype=torch.float32), requires_grad=True)

    def forward(self, x): #[72,2048,16,8]
        res = torch.zeros((x.shape[0] * x.shape[1], self.nbins)).cuda()
        inputt = x.view(x.shape[0] * x.shape[1], -1)
        for i in range(1, len(self.norm_centers)): # exclude first and last
            res[:, i] = torch.mean(self.norm(inputt, self.norm_centers[i]), 1)
        
        res[:, 0] = torch.mean(1 - self.sigmoid(inputt - self.sigmoid_semi_centers[0]), 1)
        res[:, -1] = torch.mean(self.sigmoid(inputt - self.sigmoid_semi_centers[-1]), 1)
        return res.view(x.shape[0], -1)
    
    def norm(self, x, mu):
        return (1 / (self.sigma * torch.sqrt(torch.tensor(2 * math.pi)))) * torch.exp((-0.5*((x - mu)/self.sigma)**2))

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-20*x))

class HistYusufLayer(nn.Module):
    def __init__(self, n_bins=5, inchannel=1, centers=None, width=None):
        super(HistYusufLayer, self).__init__()
        
        self.n_bins = n_bins
        self.conv_centers_inchannel = inchannel
        if centers is not None and width is not None:
            self.centers, self.width = centers, width
        else:
            self.centers, self.width = self.calc_dummy_centers_and_width()
        # count as -u
        self.conv_centers = nn.Conv2d(self.conv_centers_inchannel, \
                                      self.n_bins * self.conv_centers_inchannel, 1, \
                                      groups=self.conv_centers_inchannel, bias=True)
        self.conv_centers.weight.data.fill_(1)
        self.conv_centers.weight.requires_grad_(False)
        # initial centers
        self.conv_centers.bias.data = torch.nn.Parameter(
            -torch.tensor(np.tile(self.centers, self.conv_centers_inchannel), dtype=torch.float32))
        # count as w
        self.conv_widths = nn.Conv2d(self.n_bins * self.conv_centers_inchannel, \
                                    self.n_bins * self.conv_centers_inchannel, 1, \
                                    groups=self.n_bins * self.conv_centers_inchannel, bias=True)
        self.conv_widths.weight.data.fill_(-1)
        self.conv_widths.weight.requires_grad_(False)
        # initial width
        if type(self.width) == np.ndarray or type(self.width) == list:
            self.conv_widths.bias.data = torch.nn.Parameter(torch.tensor(np.tile(self.width, self.conv_centers_inchannel), dtype=torch.float32))
        else:
            self.conv_widths.bias.data.fill_(self.width)
        self.relu1 = nn.Threshold(threshold=1.0, value=0.0)
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    
    def forward(self, x):
        # input 2d array
#         pdb.set_trace()
        x = self.conv_centers(x)
        x = torch.abs(x)
        x = self.conv_widths(x)
        x = torch.pow(1.01, x)
        # x += 1e-6 # to accept exact value "1" in threshold (relu1)
        x = torch.add(x, 1e-6)
        x = self.relu1(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def calc_dummy_centers_and_width(self):
        bin_edges = np.linspace(-0.05, 1.05, self.n_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        return centers[:-1], (bin_edges[2] - bin_edges[1]) / 2

