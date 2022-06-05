import os
import random

import torch
import numpy as np


class CFG:
    base_path     = '/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_KL/'
    #'/kl_crowe_vit/20220511_DRR_with_Crowe_KL/20220511_OsakaHip_TwoSide_KL_Crowe.csv'
    #base_path     = 'c:\\Users\\masuda_m\\code\\20220511_DRR_with_Crowe_KL\\'
    image_path    = base_path + "DRR_AP"
    csv_path      = base_path + "20220511_OsakaHip_TwoSide_KL_Crowe.csv"
    labels_dict   = {0:'KL=0,Crowe=0',1:'KL=1,Crowe=0',2:'KL=1,Crowe=1',3:'KL=1,Crowe=2',4:'KL=1,Crowe=3',
                    5:'KL=1,Crowe=4',6:'KL=2,Crowe=4',7:'KL=3,Crowe=4',8:'KL=4,Crowe=4'}
    labels        = [0,1,2,3,4,5,6,7,8]
    labels_name   = ['0,0','1,0','1,1','1,2','1,3',
                    '1,4','2,4','3,4','4,4']
    seed          = 101
    debug         = False # set debug=False for Full Training
    test_size     = 0.3

    model_name    = 'VisionTransformer'
    batch_size    = 32
    image_size    = 224
    patch_size    = 16
    epochs        = 3
    lr            = 1e-3
    #scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-4
    #T_max         = epochs
    T_max         = int(30000/batch_size*epochs)
    #T_max         = int(30000/batch_size*epochs)
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//batch_size)
    n_fold        = 4
    folds         = [0,1,2,3]
    num_classes   = 9


    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu()

    def set_seed(seed = 42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print('> SEEDING DONE')
