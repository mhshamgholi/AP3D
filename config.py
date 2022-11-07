import numpy as np
from torchvision import transforms as torchT
from models import MyModels

class Config():

        
    def __init__(self):

        self.print_model_parameters_trainable = True
        self.print_model_layers = True
        # width = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        # width = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4])
        # centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5, 13, 21, 29, 37])
        self.centers = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        self.widths = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


        self.hist_by_prof_edges = [0, 0.2, 0.4, 0.6, 0.8, 1] # 15, 30]
        self.use_dropout = False
        self.use_hist = True #False
        self.use_just_last_bin = False
        self.concat_hist_max = True #False
        self.use_linear_to_get_important_features = True # 2048 * 8 -> 2048

        self.use_resnet18 = True
        self.last_feature_dim = 512 if self.use_resnet18 else 2048
        # use_pad_for_resnet18_Bottleneck3D = True # if use_resnet18 is False then this param will be ignored

        self.use_linear_to_merge_features = False # ( 2048 * 8 ) 8 -> 1


        self.what_to_freeze_startwith = ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'hist.'] # bn. , classifier. , feature_reduction. 

        if self.use_linear_to_merge_features and self.use_linear_to_get_important_features:
            raise Exception("both 'use_linear_to_merge_features' 'use_linear_to_get_important_features are True'")
            
        # init hist
        self.init_hist

    def init_hist(self, hist_name):
        self.hist_name = hist_name
        if self.hist_name == "HistByProf":
            self.hist_model = MyModels.HistByProf(edges=self.hist_by_prof_edges, use_just_last_bin=self.use_just_last_bin)
        elif self.hist_name == "HistYusufLayer":
            self.hist_model = MyModels.HistYusufLayer(inchannel=self.last_feature_dim, centers=self.centers, width=self.widths)
        


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

