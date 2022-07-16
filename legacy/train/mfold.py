import os
import re
import sys
import time
import copy
import gc
import random
import pickle
from glob import glob
import shutil
import joblib
import pathlib
from collections import defaultdict
from matplotlib.pyplot import axis
from numpy import matrix

import pandas as pd

from tqdm import tqdm
from dataset import TrainDataset,TestDataset
tqdm.pandas()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchvision import models
from create_fold import prepare_loaders
from select_model import choose_model
from utils.Configuration import CFG
from utils.EvaluationHelper import EvaluationHelper
from utils.VisualizeHelper import visualize_confusion_matrix, visualize_image
from trainval_one_epoch import train_one_epoch, valid_one_epoch 

import wandb

import dataset


def main():
    
    #Set seed
    CFG.set_seed(CFG.seed)

    #Prepare Dataframe
    data_df = pd.read_csv(CFG.csv_path)
    data_df["target"] = data_df["Crowe"]+data_df["KL"]#create target
    data_df["UID"] = data_df["ID"].str.extract('(.+)_')#extract ID
    #print(data_df)

    #manage filename
    file_names = []

    p = pathlib.Path(f'../datalist/k{CFG.n_fold}').glob('*.txt')
    for i in p:
        #print(i.name)
        file_names.append(f'k{CFG.n_fold}/'+i.name)
    #print(file_names)

    name = []
    for j in range(len(file_names)):
        for i in range(CFG.n_fold):
            if str(i) in file_names[j]:
                name.append(os.path.join(CFG.fold_path, file_names[j]))
    #print(name)

    #run training each fold
    best_fold = -1
    best_fold_acc = -10**9

    for fold in CFG.folds:
        print(f'#'*15)
        print(f'### Fold: {fold+1}')
        print(f'#'*15)

        #prepare dataframe for each fold
        #fold dataframe
        with open(name[i]) as f:
            line = f.read().splitlines()
        with open(name[i+CFG.n_fold]) as f:
            line2 = f.read().splitlines()
        #print(len(line),len(line2))
        valid_df = data_df[data_df['UID'].isin(line)]
        train_df = data_df[data_df['UID'].isin(line2)]
        #print(train_df.shape,valid_df.shape)

        train_dataset = TrainDataset(train_df, transform=dataset.get_transforms('train'))
        valid_dataset = TestDataset(valid_df, transform=dataset.get_transforms('valid'))

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=False, pin_memory=True)

        #create model
        model = choose_model(CFG.model_name)
        #print(model)

        #Metrics
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
        #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        #wandb
        run = wandb.init(
            project='kl_crowe-project_fold', 
            config={
                "model_name": CFG.model_name,
                "learning_rate": CFG.lr,
                "epochs": CFG.epochs,
                "batch_size": CFG.batch_size,
                "optimizer": "Adam"
            },
            entity="xxmrkn",
            name=f"{CFG.model_name}|fold-{fold+1}|dim-{CFG.image_size}**2|batch-{CFG.batch_size}|lr-{CFG.lr}",)
        wandb.watch(model, log_freq=100)

        #Confirm cuda is avalirable
        if torch.cuda.is_available():
            print(f"cuda: {torch.cuda.get_device_name}\n")

        #Training 
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())

        best_acc = 0.0
        best_acc2 = 0.0
        tmp_acc = 0.0
        best_f1 = 0.0
        best_epoch = -1
        best_epoch2 = -1
        history = defaultdict(list)
        cmat = 0
        id_list = [[] for _ in range(CFG.num_classes*2-3)]
        print(f'id list{id_list}')
            history['Train Accuracy'].append(train_acc)
            history['Train F-measure'].append(train_f1)

            history['Valid Loss'].append(valid_loss)
            history['Valid Accuracy'].append(valid_acc)
            history['Valid Accuracy2'].append(valid_acc2)
            history['Valid F-measure'].append(valid_f1)

            # Log the metrics
            wandb.log({"Train Loss": train_loss, 
                        "Valid Loss": valid_loss,
                        "Train_Accuracy": train_acc,
                        "Valid Accuracy": valid_acc,
                        "Valid Accuracy2": valid_acc2,
                        "Train F-measure": train_f1,
                        "Valid F-measure": valid_f1,
                        "LR":scheduler.get_last_lr()[0]})

            #Training results
            print(f"Train Loss: {train_loss} Train Acc: {train_acc} Train f1: {train_f1}")
            
            #Visualize Validation ConfusionMatrix
            print(cmatrix,cmatrix.shape)

            #Validation results
            print(f"Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid Acc2: {valid_acc2} Valid f1: {valid_f1}")

            #if valid_f1 > best_f1:
            #    print(f"Valid F1-Score Improved ({best_f1:0.4f} ---> {valid_f1:0.4f})")
            #    best_f1 = valid_f1
            #    best_epoch = epoch
            #    run.summary["Best F-measure"] = best_f1
            #    run.summary["Best Epoch"]   = best_epoch
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    PATH = f"best_epoch-{epoch:02d}.bin"
            #    torch.save(model.state_dict(), PATH)
            #    wandb.save(PATH)
            
            #If the score improved
            if valid_acc > best_acc:
                flag = True
                print(f"Valid Accuracy Improved ({best_acc:0.4f} ---> {valid_acc:0.4f})")

                best_acc = valid_acc
                best_epoch = epoch+1

                f = open(f"outputs/{CFG.model_name}_conf_mat.txt","wb")
                pickle.dump(cmatrix,f)
                print('--> Saved Confusion Matrix')

                run.summary["Best Accuracy"] = best_acc
                run.summary["Best Epoch"]   = best_epoch

                for i,c in zip(id_list,cnt):
                        for j in range(len(i)):
                            label = re.findall('AP\\\(.*)',i[j])#extract ID

                            actual = data_df[data_df['ID'].str.contains(*label)]['target']
                            pred = actual+c

                            actual_label = CFG.labels_dict[actual.item()]
                            pred_label = CFG.labels_dict[pred.item()]

                            path.append(i[j])
                            lab.append(*label)
                            ac.append(actual_label)
                            pre.append(pred_label)

                visualize_image(path,lab,ac,pre,others,flag)
                id_list = [[] for _ in range(CFG.num_classes-2)]

                best_model_wts = copy.deepcopy(model.state_dict())
            
            if valid_acc2 > best_acc2:
                flag = False
                print(f"Valid Accuracy2 Improved ({best_acc2:0.4f} ---> {valid_acc2:0.4f})")

                best_acc2 = valid_acc2
                best_epoch2 = epoch+1

                f = open(f"outputs/{CFG.model_name}_conf_mat2.txt","wb")
                pickle.dump(cmatrix,f)
                print('--> Saved Confusion Matrix2')

                run.summary["Best Accuracy2"] = best_acc2
                run.summary["Best Epoch2"]   = best_epoch2
                
                #extract and visualize outliers
                for i,c in zip(id_list,cnt):
                        for j in range(len(i)):
                            label = re.findall('AP\\\(.*)',i[j])#extract ID

                            actual = data_df[data_df['ID'].str.contains(*label)]['target']
                            pred = actual+c

                            actual_label = CFG.labels_dict[actual.item()]
                            pred_label = CFG.labels_dict[pred.item()]

                            path.append(i[j])
                            lab.append(*label)
                            ac.append(actual_label)
                            pre.append(pred_label)

                visualize_image(path,lab,ac,pre,others,flag)
                id_list = [[] for _ in range(CFG.num_classes-2)]


                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        #visualize confusion matrix
        #print(cmat)

        #visualize_confusion_matrix(cmat,CFG.labels_name,CFG.labels_name)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')
        print(f'Best Epoch {best_epoch}, Best val Accuracy: {best_acc:.4f}, Best Epoch {best_epoch2}, Best val Accuracy2: {best_acc2:.4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)

        run.finish()

        #display wandb webpage link
        print(f"wandb website ------> {run.url}")

        #remove wandb files
        print(os.path.isdir('wandb'))
        #shutil.rmtree("wandb")

    return model, history


if __name__ == '__main__':
    main()



