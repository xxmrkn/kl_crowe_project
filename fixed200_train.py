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
import numpy as np

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
from utils.fixed200_VisualizeHelper import visualize_image, visualize_total_image
from trainval_one_epoch import train_one_epoch, valid_one_epoch 

import wandb

import dataset


def main():
    
    #Set seed
    CFG.set_seed(CFG.seed)

    #Prepare Dataframe
    f = open(CFG.fixed200_df_path,"rb")
    data_df = pickle.load(f)
    data_df = pd.DataFrame(data_df)
    data_df['target']

    #0714ここからはじめるipadにメモってる内容を実装
    #data_df["target"] = data_df["Crowe"]+data_df["KL"]#create target

    print(data_df)

    #manage filename
    file_names = []

    p = pathlib.Path(f'../datalist2/k{CFG.n_fold}').glob('*.txt')
    for i in p:
        #print(i.name)
        file_names.append(f'k{CFG.n_fold}/'+i.name)
    print(file_names)

    name = []
    for j in range(len(file_names)):
        for i in range(CFG.n_fold):
            if str(i) in file_names[j]:
                name.append(os.path.join(CFG.fixed_fold_path, file_names[j]))
    print(name)

    #run training each fold
    best_fold = -1
    best_fold_acc = -10**9
    total_confusion_matrix = 0
    total_path = []
    total_id = []
    total_actual = []
    total_pred = []

    total_path2 = []
    total_id2 = []
    total_actual2 = []
    total_pred2 = []

    class_report_labels = []
    class_report_preds = []

    for fold in CFG.folds:
        print(f'#'*15)
        print(f'### Fold: {fold+1}')
        print(f'#'*15)

        #prepare dataframe for each fold
        #fold dataframe
        with open(name[fold]) as f:
            line = f.read().splitlines()
        with open(name[fold+CFG.n_fold]) as f:
            line2 = f.read().splitlines()
        #print(len(line),len(line2))
        valid_df = data_df[data_df['UID'].isin(line)]
        train_df = data_df[data_df['UID'].isin(line2)]
        print(len(train_df),len(valid_df))

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
            project='fixed_kl_crowe-project', 
            config={
                "model_name": CFG.model_name,
                "learning_rate": CFG.lr,
                "fold": CFG.n_fold,
                "epochs": CFG.epochs,
                "batch_size": CFG.batch_size,
                "optimizer": "Adam"
            },
            entity="xxmrkn",
            name=f"fixed200_{CFG.n_fold}fold|{CFG.model_name}|fold-{fold+1}|dim-{CFG.image_size}**2|batch-{CFG.batch_size}|lr-{CFG.lr}",)
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
        
        crnt_acc = 0.0

        
        for epoch in range(CFG.epochs):
            gc.collect()
            print(f'Epoch {epoch+1}/{CFG.epochs}')
            print('-' * 10)

            id_list = [[] for _ in range(CFG.num_classes*2-1)]
            id_list2 = [[] for _ in range(CFG.num_classes*2-1)]
            path = []
            lab = []
            ac = []
            pre = []
            path2 = []
            lab2 = []
            ac2 = []
            pre2 = []

            train_loss, train_acc, train_f1 = train_one_epoch(model, optimizer, scheduler,
                                                                criterion, train_loader,
                                                                CFG.device)
            valid_loss, valid_acc, valid_f1, cmatrix, dataset_size, id_list, id_list2, tmp_acc, clsrepo_preds, clsrepo_labels = valid_one_epoch(
                                                                                                    model, optimizer, criterion, valid_loader,
                                                                                                    CFG.device, id_list, id_list2, tmp_acc, 
                                                                                                    epoch+1, fold+1, class_report_preds, class_report_labels)

            valid_acc2, others, others2 = EvaluationHelper.one_mistake_acc(cmatrix, dataset_size)
            #othersはnomalと1neighbor以外の外れ値の数、可視化するときのsubplotの引数として渡す

            cnt = [i  for i in range(-CFG.num_classes+1,CFG.num_classes)]

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
            print(cmatrix)

            #Validation results
            print(f"Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid Acc2: {valid_acc2} Valid f1: {valid_f1}")

            if epoch+1==CFG.epochs:
                f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/confusion-matrix_normal-acc/fixed200_{CFG.model_name}_fold{fold+1}_epoch{epoch+1}{CFG.epochs}_conf_mat.txt","wb")
                pickle.dump(cmatrix,f)
                print('--> Saved Confusion Matrix')

            if epoch+1==CFG.epochs:
                f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/confusion-matrix_1neighbor-acc/fixed200_{CFG.model_name}_fold{fold+1}_epoch{epoch+1}{CFG.epochs}_conf_mat2.txt","wb")
                pickle.dump(cmatrix,f)
                print('--> Saved Confusion Matrix2')

            #print(data_df)
            #If the score improved
            if valid_acc > best_acc:
                print(f"Valid Accuracy Improved ({best_acc:0.4f} ---> {valid_acc:0.4f})")

                best_acc = valid_acc
                best_epoch = epoch+1

                run.summary["Best Accuracy"] = best_acc
                run.summary["Best Epoch"]   = best_epoch

            if epoch+1==CFG.epochs:
                for i,c in zip(id_list,cnt):
                    flag = 1 
                    if i in [0]:
                        pass
                    else:
                        for j in range(len(i)):

                            label = re.findall('AP/(.*)',i[j])#extract ID
                            #print(f'label:{label}')

                            actual = data_df[data_df['ID'].str.contains(*label)]['target']
                            pred = actual+c
                            #print(actual.item(),pred.item())

                            actual_label = CFG.labels_dict[actual.item()]
                            pred_label = CFG.labels_dict[pred.item()]
                            #print(f'actuallabel:{actual_label},predlabel:{pred_label}')
                                
                            path.append(i[j])
                            lab.append(*label)
                            ac.append(actual_label)
                            pre.append(pred_label)

                            if epoch+1==CFG.epochs:
                                total_path.append(i[j])
                                total_id.append(*label)
                                total_actual.append(actual_label)
                                total_pred.append(pred_label)

                print(f"number of outliers{others}")
                visualize_image(path,lab,ac,pre,others,flag,fold+1,epoch+1)

            best_model_wts = copy.deepcopy(model.state_dict())
            
            if valid_acc2 > best_acc2:

                print(f"Valid Accuracy2 Improved ({best_acc2:0.4f} ---> {valid_acc2:0.4f})")

                best_acc2 = valid_acc2
                best_epoch2 = epoch+1

                run.summary["Best Accuracy2"] = best_acc2
                run.summary["Best Epoch2"]   = best_epoch2
                
            if epoch+1==CFG.epochs: 
                flag = 2
                #extract and visualize outliers
                for i,c in zip(id_list2,cnt):
                    if i in [-1,0,1]:
                        pass
                    else:
                        for j in range(len(i)):
                            label = re.findall('AP/(.*)',i[j])#extract ID

                            actual = data_df[data_df['ID'].str.contains(*label)]['target']
                            pred = actual+c

                            actual_label = CFG.labels_dict[actual.item()]
                            pred_label = CFG.labels_dict[pred.item()]

                            path2.append(i[j])
                            lab2.append(*label)
                            ac2.append(actual_label)
                            pre2.append(pred_label)

                            if epoch+1==CFG.epochs:
                                total_path2.append(i[j])
                                total_id2.append(*label)
                                total_actual2.append(actual_label)
                                total_pred2.append(pred_label)

                print(f"number of outliers{others2}")
                visualize_image(path2,lab2,ac2,pre2,others2,flag,fold+1,epoch+1)
            
                best_model_wts = copy.deepcopy(model.state_dict())
            #manage confusion matrix
            if epoch+1 == CFG.epochs:
                total_confusion_matrix += cmatrix


            #calculate total accuracy
            if epoch+1==CFG.epochs and fold+1==CFG.n_fold:
                total_dataset_size = len(train_dataset)+len(valid_dataset)
                print(f'total_confusion_matrix:{total_confusion_matrix}')
                print(f"total sum:{np.sum(total_confusion_matrix)}")
                normal_acc, neighbor_acc, remain, remain2 = EvaluationHelper.total_acc(total_confusion_matrix,total_dataset_size)
                print(f"normal_acc:{normal_acc},1neighbor_acc:{neighbor_acc},remain:{remain},remain2:{remain2}")

                f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/fixed200_{CFG.model_name}_{fold+1}fold_{epoch+1}{CFG.epochs}epoch_totalconfusion_matrix.txt","wb")
                pickle.dump(total_confusion_matrix,f)
                print('--> Saved Total Confusion Matrix')
        print()
        #visualize confusion matrix
        #print(cmat)

        #Visualize outliers
        if epoch+1==CFG.epochs and fold+1==CFG.n_fold:
            sign=1
            visualize_total_image(total_path,total_id,total_actual,total_pred,remain,normal_acc,neighbor_acc,sign)
            #Visualize outliers2
            sign=2
            visualize_total_image(total_path2,total_id2,total_actual2,total_pred2,remain2,normal_acc,neighbor_acc,sign)
            
            class_report = EvaluationHelper.total_report(clsrepo_preds, clsrepo_labels)
            f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/fixed200_{CFG.model_name}_{fold+1}fold_{epoch+1}{CFG.epochs}epoch_class_report.txt","wb")
            pickle.dump(class_report,f)
            print('--> Saved Classification Report')
        #visualize_confusion_matrix(cmat,CFG.labels_name,CFG.labels_name)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')
        print(f'Best Epoch {best_epoch}, Best val Accuracy: {best_acc:.4f}, Best Epoch {best_epoch2}, Best val Accuracy2: {best_acc2:.4f}')

        #print(f"normal_acc:{normal_acc},1neighbor_acc:{neighbor_acc},remain:{remain},remain2:{remain2}")

        # load best model weights
        model.load_state_dict(best_model_wts)

        run.finish()

        #display wandb webpage link
        print(f"wandb website ------> {run.url}")

        #remove wandb files
        print(os.path.isdir('wandb'))
        #shutil.rmtree("wandb")

    return model


if __name__ == '__main__':
    main()
