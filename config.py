import numpy as np
from torchvision import transforms as torchT

width = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4])
centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5, 13, 21, 29, 37])
nbins = len(centers)

hist_by_prof_edges = [0, 0.2, 0.4, 0.6, 0.8, 1] # 15, 30]
use_dropout = False
use_hist = True #False
use_just_last_bin = True
concat_hist_max = True #False

use_resnet18 = False
last_feature_dim = 512 if use_resnet18 else 2048
use_pad_for_resnet18_Bottleneck3D = True # if use_resnet18 is False then this param will be ignored


def get_spatial_transform_train(args):
#     return torchT.Compose([
#                 torchT.Scale((int(args.height * 1.2), int(args.width * 1.2)), interpolation=3), #(args.height, args.width)
#                 torchT.RandomCrop((args.height, args.width)),
#                 torchT.ColorJitter(brightness=.4), #torchT.RandomRotation(degrees=(-20, 20)),
#                 torchT.RandomHorizontalFlip(p=0.5),
#                 torchT.ToTensor(),
#                 torchT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
    return torchT.Compose([
                torchT.Scale((int(args.height), int(args.width)), interpolation=3), #(args.height, args.width)
                torchT.RandomHorizontalFlip(p=0.5),
                torchT.ToTensor(),
                torchT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])