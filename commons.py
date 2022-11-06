import torch
import config

def modify_model(model, args, conf: config.Config):
#     if args.arch == "ap3dres50" and args.pretrain != "":
#         return None
    
    map_loc = 'cuda' if torch.cuda.is_available() else "cpu"
    if args.pretrain != '':
        my_state_dict = torch.load(args.pretrain, map_location=map_loc)['state_dict']
        # copy params from random model to pretrain model, because some layer are new or some layer's size is changed
        for n, p in model.named_parameters():
            if ('hist.' in n) or ('classifier' in n):
                my_state_dict[n] = p
        my_state_dict = dict(filter(lambda elem: 'bn.' not in elem[0], my_state_dict.items()))

        model.load_state_dict(my_state_dict, strict=False)
        # model.bn.load_state_dict(my_state_dict['bn'])

    for n, p in model.named_parameters():
        if any(str(n).startswith(param) for param in conf.what_to_freeze_startwith):
            p.requires_grad = False
        else:
            p.requires_grad = True
    
    print("pretrain state dict loaded")
    # exit()
    for name, param in model.named_parameters():
        print(f'>>> module {name} is trainable ? {param.requires_grad}')
    print('-'*10)
    print('model layers:')
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    

def log_model_after_epoch(model):
#     print(f'model hist : edges : {model.module.hist.hist_edges}')
    pass