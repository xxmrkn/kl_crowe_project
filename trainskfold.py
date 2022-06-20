import os
import sys
import time
import copy
import gc
import random
import pickle
from glob import glob
import shutil
import joblib
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
from create_fold import prepare_loaders

from utils.Configuration import CFG
from utils.EvaluationHelper import EvaluationHelper
from utils.VisualizeHelper import visualize_confusion_matrix
from trainval_one_epoch import train_one_epoch, valid_one_epoch

from select_model import choose_model
import wandb
import dataset


def main():
    
    #Set seed
    CFG.set_seed(CFG.seed)

    #Prepare Dataframe
    print(CFG.csv_path)
    data_df = pd.read_csv(CFG.csv_path)
    data_df["target"] = data_df["Crowe"]+data_df["KL"]

    #Stratified Kfold
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    #create fold
    for fold, (tr_idx, va_idx) in enumerate(skf.split(data_df, data_df["target"])):
        data_df.loc[va_idx, 'fold'] = fold
    print(data_df)

    #check dataframe and count target
    for i in range(4):
        for j in range(9):
            print(f'number of target {j} in fold:{i} dataframe')
            print(i,(data_df['target']!=j).sum(),(data_df['target']==j).sum())

    #run training each fold
    best_fold = -1
    best_fold_acc = -10**9

    for fold in CFG.folds:
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)

        #create model
        model = choose_model(CFG.model_name)

        #prepare loader 
        train_loader, valid_loader = prepare_loaders(fold=fold,data_df=data_df)

        #Metrics
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
        #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        #wandb
        run = wandb.init(
            project='vit_kl_crowe-project_stkfold', 
            config={
                "learning_rate": CFG.lr,
                "epochs": CFG.epochs,
                "batch_size": CFG.batch_size,
                "optimizer": "Adam"
            },
            entity="xxmrkn",
            name=f"fold-{fold}|dim-{CFG.image_size}x{CFG.image_size}|model-{CFG.model_name}",)
        wandb.watch(model, log_freq=100)

        #Confirm cuda is avalirable
        if torch.cuda.is_available():
            print(f"cuda: {torch.cuda.get_device_name}\n")

        #Training 
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_acc2 = 0.0
        best_f1 = 0.0
        best_epoch = -1
        best_epoch2 = -1
        history = defaultdict(list)
        cmat = 0
        
        for epoch in range(CFG.epochs):
            gc.collect()
            print(f'Epoch {epoch+1}/{CFG.epochs}')
            print('-' * 10)

            train_loss, train_acc, train_f1 = train_one_epoch(model, optimizer, scheduler,
                                                                criterion, train_loader,
                                                                CFG.device)
            valid_loss, valid_acc, valid_f1, cmatrix, dataset_size = valid_one_epoch(model, optimizer, criterion, valid_loader,
                                                                CFG.device)

            #manage confusion matrix
            #cmat += cmatrix
            valid_acc2 = EvaluationHelper.one_mistake_acc(cmatrix, dataset_size)

            history['Train Loss'].append(train_loss)
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
                print(f"Valid Accuracy Improved ({best_acc:0.4f} ---> {valid_acc:0.4f})")
                best_acc = valid_acc
                best_epoch = epoch+1
                f = open("utils/conf_mat.txt","wb")
                pickle.dump(cmatrix,f)
                run.summary["Best Accuracy"] = best_acc
                run.summary["Best Epoch"]   = best_epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if valid_acc2 > best_acc2:
                print(f"Valid Accuracy2 Improved ({best_acc2:0.4f} ---> {valid_acc2:0.4f})")
                best_acc2 = valid_acc2
                best_epoch2 = epoch+1
                f = open("utils/conf_mat2.txt","wb")
                pickle.dump(cmatrix,f)
                run.summary["Best Accuracy2"] = best_acc2
                run.summary["Best Epoch2"]   = best_epoch2
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