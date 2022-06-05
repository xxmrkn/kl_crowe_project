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
from numpy import matrix

import pandas as pd

from tqdm import tqdm
from dataset import TrainDataset,TestDataset
tqdm.pandas()
from sklearn.model_selection import train_test_split
from model import ViT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torchvision import models

from utils.Configuration import CFG
from utils.EvaluationHelper import EvaluationHelper
from utils.VisualizeHelper import visualize_confusion_matrix
from trainval_one_epoch import train_one_epoch, valid_one_epoch 

import wandb

import dataset


def main():

    #Define Pretrained ViT model
    model = models.vit_l_16(pretrained=True)
    model.heads = nn.Sequential(
        nn.Linear(
        in_features=1024,
        out_features=9
    ))
    model = model.to(CFG.device)

    #View Model Detail
    print(model)
    
    #Set seed
    CFG.set_seed(CFG.seed)

    #Prepare Dataframe
    data_df = pd.read_csv(CFG.csv_path)
    data_df["target"] = data_df["Crowe"]+data_df["KL"]

    #Training and Validation
    train_df,valid_df = train_test_split(data_df,test_size=CFG.test_size)

    #Prepare Dataset
    train_dataset = TrainDataset(train_df, transform=dataset.get_transforms('train'))
    valid_dataset = TestDataset(valid_df, transform=dataset.get_transforms('valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                            num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                            num_workers=4, shuffle=False, pin_memory=True)

    #Metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #wandb
    run = wandb.init(
        project='vit_kl_crowe-project_dataaug', 
        config={
            "learning_rate": CFG.lr,
            "epochs": CFG.epochs,
            "batch_size": CFG.batch_size,
            "patch_size": CFG.patch_size,
            "optimizer": "Adam"
        },
        entity="xxmrkn")
    wandb.watch(model, log_freq=100)

    #Confirm cuda is avalirable
    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name}\n")

    #Training 
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = -1
    history = defaultdict(list)
    cmat = 0
    
    for epoch in range(CFG.epochs):
        gc.collect()
        print(f'Epoch {epoch+1}/{CFG.epochs}')
        print('-' * 10)

        train_loss, train_acc, train_f1 = train_one_epoch(model, optimizer, scheduler,
                                                            criterion, train_loader,
                                                            CFG.device)
        valid_loss, valid_acc, valid_f1, cmatrix = valid_one_epoch(model, optimizer, criterion, valid_loader,
                                                            CFG.device)

        #manage confusion matrix
        cmat += cmatrix

        history['Train Loss'].append(train_loss)
        history['Train Accuracy'].append(train_acc)
        history['Train F-measure'].append(train_f1)

        history['Valid Loss'].append(valid_loss)
        history['Valid Accuracy'].append(valid_acc)
        history['Valid F-measure'].append(valid_f1)

            # Log the metrics
        wandb.log({"Train Loss": train_loss, 
                    "Valid Loss": valid_loss,
                    "Train_Accuracy": train_acc,
                    "Valid Accuracy": valid_acc,
                    "Train F-measure": train_f1,
                    "Valid F-measure": valid_f1,
                    "LR":scheduler.get_last_lr()[0]})

        #visualize_confusion_matrix(cmat,CFG.labels_name,CFG.labels_name)

        print(f"Train Loss: {train_loss} Train Acc: {train_acc} Train f1: {train_f1}")
        print(f"Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid f1: {valid_f1}")

        print(cmat)


        
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

        if valid_acc > best_acc:
            print(f"Valid Accuracy Improved ({best_acc:0.4f} ---> {valid_acc:0.4f})")
            best_acc = valid_acc
            best_epoch = epoch
            f = open("utils/conf_mat.txt","wb")
            pickle.dump(cmat,f)
            run.summary["Best Accuracy"] = best_acc
            run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        cmat = 0

    print()
    #visualize confusion matrix
    #print(cmat)

    #visualize_confusion_matrix(cmat,CFG.labels_name,CFG.labels_name)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')
    print(f'Best val Accuracy: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    run.finish()

    #display wandb webpage link
    print(f"wandb website ------> {run.url}")

    #remove wandb files
    print(os.path.isdir('wandb'))
    shutil.rmtree("wandb")

    return model, history


if __name__ == '__main__':
    main()