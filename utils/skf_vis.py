#%%
from re import T
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

#%%
class CFG:
    base_path ='c:\\Users\\masuda_m\\code\\20220511_DRR_with_Crowe_KL\\'
    csv_path = base_path + "20220511_OsakaHip_TwoSide_KL_Crowe.csv"
    label = ['0,0','1,0','1,1','1,2','1,3','1,4','2,4','3,4','4,4']

#%%
data_df = pd.read_csv(CFG.csv_path)
data_df['target'] = data_df['KL']+data_df["Crowe"]
data_df

#Stratified Kfold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
data_df

#create fold
fold_tr = ['fold0_tr','fold1_tr','fold1_tr','fold1_tr']
fold_va = ['fold0_va','fold1_va','fold1_va','fold1_va']

TRAIN = [[] for _ in range(4)]
VALID = [[] for _ in range(4)]

for fold, (tr_idx, va_idx) in enumerate(skf.split(data_df, data_df["target"])):

    fold_tr[fold] = data_df.iloc[tr_idx]
    fold_va[fold] = data_df.iloc[va_idx]
    for j in range(9):
        s = fold_tr[fold]['target']==j
        t = fold_va[fold]['target']==j
        TRAIN[fold].append(s.sum())
        VALID[fold].append(t.sum())

# %%
TRAIN,VALID

# %%
R=[1,2,3,4,5,6,7,8,9]

# %%
for i in range(4):
    plt.figure(figsize=(50,30))
    plt.title(f'Fold:{i} Crowe,KL', fontsize=60)
    plt.bar(R, TRAIN[i], align="edge", width=0.3, tick_label=CFG.label,label='training')
    plt.bar(R, VALID[i], align="edge", width=-0.3, tick_label=CFG.label,label='validation')
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.legend(fontsize=100)
    plt.show()
# %%
