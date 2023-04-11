import numpy as np
from torchvision import transforms as torchT
from models import MyModels
import torch
import os

class Config():

        
    def __init__(self):

        self.print_model_parameters_trainable = True
        self.print_model_layers = True
        self.print_hist_params_bool = True
        # width = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        # width = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4])
        self.widths = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.8])
        # centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5, 13, 21, 29, 37])
        self.centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 3])
        # self.centers = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        # self.widths = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


        self.hist_by_prof_edges = [0, 0.2, 0.4, 0.6, 0.8, 1] # 15, 30]
        # self.hist_by_prof_edges = [0.125, 0.375, 0.625, 0.875] # 15, 30]
        self.use_dropout = False
        self.use_hist_and_max_seprately = False # here histogram is just used for help model to train better. it is not used at inference and Xent 
        self.use_hist = True or self.use_hist_and_max_seprately #False
        self.use_just_last_bin = False
        self.concat_hist_max = False #False
        self.use_linear_to_get_important_features = False # 2048 * 8 -> 2048
        
        if self.concat_hist_max and self.use_hist_and_max_seprately:
            raise Exception("both 'concat_hist_max' 'use_hist_and_max_seprately are True'")
            

        self.use_resnet18 = True
        self.last_feature_dim = 512 if self.use_resnet18 else 2048
        # use_pad_for_resnet18_Bottleneck3D = True # if use_resnet18 is False then this param will be ignored

        self.use_linear_to_merge_features = False # ( 2048 * 8 ) 8 -> 1


        self.what_to_freeze_startwith = ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']# , , 'hist.'  # bn. , classifier. , feature_reduction. 

        if self.use_linear_to_merge_features and self.use_linear_to_get_important_features:
            raise Exception("both 'use_linear_to_merge_features' 'use_linear_to_get_important_features are True'")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # init hist 
        self.init_hist("HistByProfDiffMultiChannel")

    def init_hist(self, hist_name):
        self.hist_name = hist_name
        if self.hist_name == "HistByProf":
            self.hist_model = MyModels.HistByProf(init_edges=self.hist_by_prof_edges, use_just_last_bin=self.use_just_last_bin)
        elif self.hist_name == "HistYusufLayer":
            # self.hist_model = MyModels.HistYusufLayer(inchannel=self.last_feature_dim, centers=self.centers, width=self.widths)
            self.hist_model = MyModels.HistYusufLayer(inchannel=1, centers=self.centers, width=self.widths)
        elif self.hist_name == "HistByProfMultiChannel":
            self.hist_model = MyModels.HistByProfMultiChannel(num_channels=self.last_feature_dim, init_edges=self.hist_by_prof_edges, use_just_last_bin=self.use_just_last_bin)
        elif self.hist_name == "HistByProfDiffMultiChannel":
            self.hist_model = MyModels.HistByProfDiffMultiChannel(num_channels=self.last_feature_dim, init_edges=self.hist_by_prof_edges, use_just_last_bin=self.use_just_last_bin)
        else:
            raise Exception(f"hist_name {self.hist_name} is not supported")

        # self.hist_model = self.hist_model.to(self.device)
        
    def print_hist_params(self, epoch=None, log_path=None):
        if self.use_hist and hasattr(self, 'hist_name'):
            if self.hist_name == "HistByProf":
                l = self.hist_model.hist_edges.detach().cpu().numpy().tolist()
                l = [round(i, 6) for i in l]
                print('hist edges', l)
            elif self.hist_name == "HistYusufLayer":
                l = self.hist_model.conv_centers.bias.detach().cpu().numpy().tolist()
                l = [round(i, 6) for i in l]
                print('hist centers', l)
                l = self.hist_model.conv_widths.bias.detach().cpu().numpy().tolist()
                l = [round(i, 6) for i in l]
                print('hist widths', l)
            elif self.hist_name == "HistByProfMultiChannel" or self.hist_name == "HistByProfDiffMultiChannel":
                path = os.path.join(log_path, f'HisEdEp{str(epoch).zfill(3)}.txt')
                with open(path, 'w') as f:
                    for ii, edge in enumerate(self.hist_model.hist_edges):
                        l = edge.detach().cpu().numpy().tolist()
                        l = [round(i, 6) for i in l]
                        f.write(f'#{ii}: {l}\n')
                print(f'edges of {self.hist_name} in epoch {epoch} was writed in {path}')

            else:
                raise Exception(f"hist_name {self.hist_name} is unknow in 'print_hist_params'")

    def get_spatial_transform_train(self, args):
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

