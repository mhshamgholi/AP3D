from __future__ import absolute_import

import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from models import inflate
from models import AP3D
from models import NonLocal

from .MyModels import HistByNorm, HistYusufLayer, HistByProf
import config as conf
import pdb
import config as conf

__all__ = ['AP3DResNet50', 'AP3DNLResNet50']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)      


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super(Bottleneck3D, self).__init__()
        
        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        if hasattr(bottleneck2d, 'conv3'):
            self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
            self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if hasattr(self, 'conv3'):
            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
# #         if residual.shape[-1] == 8 and out.shape[-1] == 4 and residual.shape[-2] == 16 and out.shape[-2] == 8:
#         if conf.use_resnet18 and conf.use_pad_for_resnet18_Bottleneck3D:
#             #out torch.Size([14, 512, 4, 8, 4]) residual torch.Size([14, 512, 4, 16, 8])
#             p2d = (1, 2, 2, 4) # pad last dim by (1, 1) and 2nd to last by (2, 2)
#             temp = torch.zeros_like(residual)
#             temp[:,:, :, :out.shape[-2], :out.shape[-1]] = out
#             out = temp

#         out += residual
        out = out + residual
        out = self.relu(out)
        
        return out


class ResNet503D(nn.Module):
    def __init__(self, num_classes, block, c3d_idx, nl_idx, temperature=4, contrastive_att=True, **kwargs):
        super(ResNet503D, self).__init__()

        self.block = block
        self.temperature = temperature
        self.contrastive_att = contrastive_att
        if conf.use_resnet18:
            resnet2d = torchvision.models.resnet18(pretrained=True)
        else:
            resnet2d = torchvision.models.resnet50(pretrained=True)

        resnet2d.layer4[0].conv2.stride=(1, 1)
        if not conf.use_resnet18:
            resnet2d.layer4[0].downsample[0].stride=(1, 1)
            

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)


#         self.hist = HistYusufLayer(conf.centers, conf.width)
#         self.hist = HistYusufLayer(n_bins=conf.nbins, inchannel=conf.last_feature_dim, centers=conf.centers, width=conf.width)
        if conf.use_hist:
            # self.hist = HistByProf(edges=conf.hist_by_prof_edges)
            self.hist = HistYusufLayer(inchannel=conf.last_feature_dim, centers=conf.centers, width=conf.width)

            
        if conf.use_linear_to_merge_features and conf.use_hist:
            if not conf.use_just_last_bin:
                in_channel = 1 + self.hist.nbins if conf.concat_hist_max else self.hist.nbins
            else:
                in_channel = 1 + 1 if conf.concat_hist_max else 1

            self.linear_merge_features = nn.Linear(in_channel, 1)
        
        if conf.use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        
        if conf.use_hist:
            if conf.use_linear_to_merge_features:
                _coef = 1
            elif conf.use_just_last_bin:
                _coef = (1 + 1) if conf.concat_hist_max else 1
            else:
                _coef = (self.hist.nbins + 1) if conf.concat_hist_max else (self.hist.nbins)

            if conf.use_linear_to_get_important_features:
                # if this is true then _coef is must be one but we need the value later so we don't assinged it to 1
                self.bn = nn.BatchNorm1d(conf.last_feature_dim)
            else:
                self.bn = nn.BatchNorm1d(conf.last_feature_dim * _coef)
        else:
            self.bn = nn.BatchNorm1d(conf.last_feature_dim)
        self.bn.apply(weights_init_kaiming)

        if conf.use_hist:
            if conf.use_linear_to_merge_features:
                _coef = 1
            elif conf.use_just_last_bin:
                _coef = (1 + 1) if conf.concat_hist_max else 1
            else:
                _coef = (self.hist.nbins + 1) if conf.concat_hist_max else (self.hist.nbins)
                
#             self.classifier = nn.Linear(conf.last_feature_dim * (self.hist.nbins + 1), num_classes)
            if conf.use_linear_to_get_important_features:
                # if this is true then _coef is must be one but we need the value later so we don't assinged it to 1
                self.classifier = nn.Linear(conf.last_feature_dim, num_classes)
            else:
                self.classifier = nn.Linear(conf.last_feature_dim * (_coef), num_classes)
               
        else:
            self.classifier = nn.Linear(conf.last_feature_dim, num_classes)

        if conf.use_hist and conf.use_linear_to_get_important_features:
            self.feature_reduction = nn.Sequential(
                nn.Linear(conf.last_feature_dim * (_coef), conf.last_feature_dim),
                # nn.BatchNorm1d(conf.last_feature_dim),
                # nn.ReLU()
            )

        self.classifier.apply(weights_init_classifier)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, AP3D.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = NonLocal.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        if conf.use_dropout:
            x = F.dropout(x, p=0.25)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if conf.use_dropout:
            x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if conf.use_dropout:
            x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if conf.use_dropout:
            x = self.dropout(x)
            
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        if conf.use_hist and conf.concat_hist_max:
            x1 = self.hist(x)
            x2 = F.max_pool2d(x, x.size()[2:]).view(b*t, conf.last_feature_dim, 1) # -> [80, conf.last_feature_dim, 1, 1]
            #x = torch.cat((x1, x2), 1)
            x = torch.cat((x1, x2), 2)
        elif conf.use_hist and not conf.concat_hist_max:
            x = self.hist(x)
        else:
            x = F.max_pool2d(x, x.size()[2:])
        
        if conf.use_linear_to_merge_features and conf.use_hist:
            x = x.view(b * t * c, -1)
            x = self.linear_merge_features(x)
        
        if hasattr(self, 'feature_reduction'): # 2048 * 8 -> 2048
            
            x = x.view(b * t, -1)
            x = self.feature_reduction(x)
        
        x = x.view(b, t, -1)
        
        if not self.training:
            return x

        x = x.mean(1)
        f = self.bn(x)

        y = self.classifier(f)

        return y, f


def AP3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)


def AP3DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)
