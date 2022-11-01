import numpy as np
from torchvision import transforms as torchT

width = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4])
centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5, 13, 21, 29, 37])
nbins = len(centers)

hist_by_prof_edges = [0, 0.2, 0.4, 0.6, 0.8, 1] # 15, 30]
use_dropout = False
use_hist = True #False
use_just_last_bin = False
concat_hist_max = True #False
use_linear_to_get_important_features = True # 2048 * 8 -> 2048

use_resnet18 = True
last_feature_dim = 512 if use_resnet18 else 2048
# use_pad_for_resnet18_Bottleneck3D = True # if use_resnet18 is False then this param will be ignored

use_linear_to_merge_features = False # ( 2048 * 8 ) 8 -> 1





if use_linear_to_merge_features and use_linear_to_get_important_features:
    raise Exception("both 'use_linear_to_merge_features' 'use_linear_to_get_important_features are True'")



def get_spatial_transform_train(args):
#     return torchT.Compose([
# #                 torchT.Scale((int(args.height * 1.2), int(args.width * 1.2)), interpolation=3), #(args.height, args.width)
# #                 torchT.RandomCrop((args.height, args.width)),
#                 torchT.Scale((int(args.height), int(args.width)), interpolation=3), #(args.height, args.width)
#                 torchT.ColorJitter(brightness=.4), 
#                 torchT.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
#                 torchT.RandomRotation(degrees=(-15, 15)),
#                 torchT.RandomHorizontalFlip(p=0.5),
#                 torchT.ToTensor(),
#                 torchT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ## @#$%^&*()_
#                 torchT.RandomErasing(p=0.8, scale=(0.1, 0.2), ratio=(0.3, 3.3)),
#                 ## @#$%^&*()_
#             ])
    return torchT.Compose([
                torchT.Scale((int(args.height), int(args.width)), interpolation=3), #(args.height, args.width)
                torchT.RandomHorizontalFlip(p=0.5),
                torchT.ToTensor(),
                torchT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])