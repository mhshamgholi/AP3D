import torch
from torch import nn
import math
import numpy as np
import pdb
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    def __init__(self, edges, use_just_last_bin):
        super(HistByProf, self).__init__()
        self.hist_edges = nn.Parameter(torch.tensor(edges, dtype=torch.float32), requires_grad=True)
        self.use_just_last_bin = use_just_last_bin
#         self.norm_centers = []
#         self.sigma = 0.39
        self.nbins = len(edges) + 1
        
#         for i in range(len(edges)-1):
#             self.norm_centers.append((self.hist_edges[i] + self.hist_edges[i+1])/2)
        
#         self.norm_centers = nn.Parameter(torch.tensor(self.norm_centers, dtype=torch.float32), requires_grad=True)
#         self.sigmoid_semi_centers = nn.Parameter(torch.tensor([self.hist_edges[0], self.hist_edges[-1]], dtype=torch.float32), requires_grad=True)
        

    def forward(self, x): #[72,2048,16,8]
        
        res = torch.zeros((x.shape[0] * x.shape[1], self.nbins)).to(device)
        inputt = x.view(x.shape[0] * x.shape[1], -1)
#         for i in range(1, len(self.norm_centers)): # exclude first and last        
        for i in range(1, len(self.hist_edges)):
            res[:, i] = torch.sum(self.norm(inputt, (self.hist_edges[i-1] + self.hist_edges[i])/2, (self.hist_edges[i-1] - self.hist_edges[i])/3), 1)
            # res[:, i] = torch.sum(self.norm(inputt, (self.hist_edges[i] + self.hist_edges[i+1])/2, (self.hist_edges[i+1] - self.hist_edges[i])/4), 1)
        
        res[:, 0] = torch.sum(self.norm(inputt, self.hist_edges[0], (self.hist_edges[0] - self.hist_edges[1])/3), 1)
        # it seems that sigmoid can't count zero well.
        # res[:, 0] = torch.sum(1 - self.sigmoid(inputt - self.hist_edges[0]), 1)
        res[:, -1] = torch.sum(self.sigmoid(inputt - self.hist_edges[-1]), 1)
        # res[:, -1] = torch.sum(self.sigmoid(inputt - self.hist_edges[-1]), 1)
        if self.use_just_last_bin:
            res = res[:, -1] # get last bin
        res = res.view(x.shape[0], x.shape[1], -1)
#         res = res * x.shape[-1] * x.shape[-2] # unnormalize to prevent from gradient vannish
        return res # [72,2048,7]
    
    def norm(self, x, mu, sigma):
        return torch.exp((-0.5*((x - mu)/sigma)**2))

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-20*x))

class HistYusufLayer(nn.Module):
    def __init__(self, inchannel=1, centers=None, width=None):
        super(HistYusufLayer, self).__init__()
        
        
        self.conv_centers_inchannel = inchannel
        if centers is not None and width is not None:
            self.centers, self.width = centers, width
        else:
            self.centers, self.width = self.calc_dummy_centers_and_width()

        self.nbins = len(self.centers)
        # count as -u
        self.conv_centers = nn.Conv2d(self.conv_centers_inchannel, \
                                      self.nbins * self.conv_centers_inchannel, 1, \
                                      groups=self.conv_centers_inchannel, bias=True)
        self.conv_centers.weight.data.fill_(1)
        self.conv_centers.weight.requires_grad_(False)
        # initial centers
        self.conv_centers.bias.data = torch.nn.Parameter(
            -torch.tensor(np.tile(self.centers, self.conv_centers_inchannel), dtype=torch.float32))
        # count as w
        self.conv_widths = nn.Conv2d(self.nbins * self.conv_centers_inchannel, \
                                    self.nbins * self.conv_centers_inchannel, 1, \
                                    groups=self.nbins * self.conv_centers_inchannel, bias=True)
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
        bt, c, h, w = x.shape
        if self.conv_centers_inchannel == 1:
            x = x.view(bt * c, 1, h, w)
        x = self.conv_centers(x)
        x = torch.abs(x)
        x = self.conv_widths(x)
        x = torch.pow(1.01, x)
        # x += 1e-6 # to accept exact value "1" in threshold (relu1)
        x = torch.add(x, 1e-6)
        x = self.relu1(x)
        x = self.gap(x)
        x = x.view(bt, c, self.nbins)
        # x = torch.flatten(x, start_dim=1)
        return x
    
    def calc_dummy_centers_and_width(self):
        bin_edges = np.linspace(-0.05, 1.05, self.nbins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        return centers[:-1], (bin_edges[2] - bin_edges[1]) / 2

