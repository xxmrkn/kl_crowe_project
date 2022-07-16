#%% 
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CFG:
    base_path     = '/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_KL/'
    fold_path     = '/win/salmon/user/masuda/project/datalist/'
    #'/kl_crowe_vit/20220511_DRR_with_Crowe_KL/20220511_OsakaHip_TwoSide_KL_Crowe.csv'
    #base_path     = 'c:\\Users\\masuda_m\\code\\20220511_DRR_with_Crowe_KL\\'
    image_path    = base_path + "DRR_AP"
    csv_path      = base_path + "20220511_OsakaHip_TwoSide_KL_Crowe.csv"
    labels_dict   = {0:'KL=0,Crowe=0',1:'KL=0,Crowe=1',2:'KL=1,Crowe=1',3:'KL=2,Crowe=1',4:'KL=3,Crowe=1',
                    5:'KL=4,Crowe=1',6:'KL=4,Crowe=2',7:'KL=4,Crowe=3',8:'KL=4,Crowe=4'}
    labels        = [0,1,2,3,4,5,6,7,8]
    labels_name   = ['0,0','1,0','1,1','1,2','1,3',
                    '1,4','2,4','3,4','4,4']
    seed          = 101
    debug         = False # set debug=False for Full Training
    test_size     = 0.3

    # model name
    # 'VisionTransformer_Base16','VisionTransformer_Base32'
    # 'VisionTransformer_Large16','VisionTransformer_Large32'
    # 'VGG16','Inceptionv3'
    # 'efficientnet_b0','efficientnet_b3','efficientnet_b7'

    model_name    = 'VisionTransformer_Base16'
    batch_size    = 32
    image_size    = 224
    patch_size    = 16
    epochs        = 30
    lr            = 1e-5
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    #T_max         = epochs
    #T_max         = 2
    T_max         = int(30000/batch_size*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-4
    n_accumulate  = max(1, 32//batch_size)
    n_fold        = 4
    #folds         = [0,1,2,3]
    folds          = [i for i in range(n_fold)]
    num_classes   = 9
#%%

def acc(matrix,dataset_size):
        taikaku1 = sum(np.diag(matrix)) #対角成分
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1)) #対角成分の両サイド
        other1 = dataset_size-taikaku1#normal
        other2 = dataset_size-taikaku1-taikaku2 #1 neighbor
        # print(taikaku1,taikaku2)
        print(f'taikaku1:{taikaku1},taikaku2:{taikaku2},taikaku_total:{taikaku1+taikaku2}')
        print(f'dataset_size:{dataset_size},other1:{other1},other2:{other2}')
        return taikaku1/dataset_size,(taikaku1+taikaku2)/dataset_size,taikaku1,taikaku1+taikaku2

f = open(r"C:\\Users\\masuda_m\\code\\VisionTransformer_Base16_4fold_3030_totalconfusion_matrix.txt","rb")
matrix = pickle.load(f)
normal_acc, neighbor_acc, taikaku1,taikaku2 = acc(matrix,np.sum(matrix))

print()

print(f'normal_acc:{normal_acc},neighbor_acc:{neighbor_acc}')
print(f'taikaku1:{taikaku1}, taikaku2:{taikaku2}')



#%%
f = open(r"C:\\Users\\masuda_m\\code\\VisionTransformer_Base16_4fold_3030_totalconfusion_matrix.txt","rb")
list_row = pickle.load(f)
cm = pd.DataFrame(data=list_row, index=['KL=0,Crowe=0','KL=0,Crowe=1','KL=1,Crowe=1','KL=2,Crowe=1','KL=3,Crowe=1',
                    'KL=4,Crowe=1','KL=4,Crowe=2','KL=4,Crowe=3','KL=4,Crowe=4'], 
                    columns=['KL=0,Crowe=0','KL=0,Crowe=1','KL=1,Crowe=1','KL=2,Crowe=1','KL=3,Crowe=1',
                    'KL=4,Crowe=1','KL=4,Crowe=2','KL=4,Crowe=3','KL=4,Crowe=4'])

# %%
plt.figure(figsize=(12, 12))

sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues',fmt='d',vmax=200, vmin=0, center=120)
#sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='GnBu',fmt='d',vmax=40, vmin=-10, center=0)
#sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='BuPu',fmt='d',vmax=40, vmin=-10, center=0)
plt.title(f'{CFG.model_name}   {CFG.n_fold}Fold  {CFG.epochs}Epoch\nNormal-Acc: {normal_acc:.4f}  ({taikaku1}/{np.sum(matrix)})\n1Neighbor-Acc: {neighbor_acc:.4f}  ({taikaku2}/{np.sum(matrix)})')
plt.yticks(rotation=0)
plt.xticks()
plt.xlabel("Pred", fontsize=13, rotation=0)
plt.ylabel("True", fontsize=13)
file_name = f"experiment_fig/{CFG.model_name}/{CFG.n_fold}fold/{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_confusion_matrix.png"
plt.savefig(file_name)

# %%
