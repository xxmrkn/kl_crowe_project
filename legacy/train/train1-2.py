#自分がもともと使っていたベースライン
import os
import sys
import time
import copy
import gc
import random
from glob import glob
import shutil
import joblib
from collections import defaultdict

import pandas as pd
import numpy as np

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

from utils.Configuration import CFG
from utils.EvaluationHelper import EvaluationHelper
from utils.VisualizeHelper import visualize_plot_loss,visualize_plot_acc,visualize_plot_f1
import wandb

import dataset


def main():

    model = ViT(
    image_size = CFG.image_size, #256*256->65,536
    patch_size = CFG.patch_size,  #32*32->1,024
    #64 patchs -> 8patchs * 8patchs
    num_classes = CFG.num_classes,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
    ).to(CFG.device,dtype=torch.float32)
    

    CFG.set_seed(CFG.seed)
    #print(CFG.csv_path)

    data_df = pd.read_csv(CFG.csv_path)
    data_df["target"] = data_df["Crowe"]+data_df["KL"]

    train_df,valid_df = train_test_split(data_df,test_size=0.3)

    train_dataset = TrainDataset(train_df, transform=dataset.get_transforms('train'))
    valid_dataset = TestDataset(valid_df, transform=dataset.get_transforms('valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=False, pin_memory=True)

    datasets = {'train': train_dataset,
            'valid': valid_dataset}

    dataloaders = {'train': train_loader,
                'valid': valid_loader}

    #Metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
    num_epochs = CFG.epochs
    grad_scaler = torch.cuda.amp.GradScaler()

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = -1

    #wandb
    run = wandb.init(
        project='vit_kl_crowe-project', 
        config={
            "learning_rate": CFG.lr,
            "epochs": CFG.epochs,
            "batch_size": CFG.batch_size,
            "optimizer": "Adam"
        },
        entity="xxmrkn")
    wandb.watch(model, log_freq=100)
    
    for epoch in range(CFG.epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_acc = 0.0
            running_f1 = 0.0
            
            pbar = tqdm(enumerate(dataloaders[phase],start=1), total=len(dataloaders[phase]))
            
            for batch, (inputs,labels) in pbar:
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)
                #print(inputs, labels)
                
                # Zero out the grads
                optimizer.zero_grad()
                
                # Forward
                # Track history in train mode
                #with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast(True if phase=='train' else False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    acc =  EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
                    f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                
                running_loss += loss.item()
                running_acc += acc
                running_f1 += f_m
                
            epoch_loss = running_loss / batch
            epoch_acc = running_acc / batch
            epoch_f1 = running_f1 / batch

            # Log the metrics
            if phase=='train':
                wandb.log({"Train Loss": epoch_loss, 
                        "Train Accuracy": epoch_acc,
                        "Train F-measure": epoch_f1,
                        "LR":scheduler.get_last_lr()[0]})
            if phase=='valid':
                wandb.log({"Valid Loss": epoch_loss, 
                        "Valid Accuracy": epoch_acc,
                        "Valid F-measure": epoch_f1,
                        "LR":scheduler.get_last_lr()[0]})
            
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))
            
            if phase == 'valid' and epoch_f1 > best_f1:
                print(f"Valid Score Improved ({best_f1:0.4f} ---> {epoch_f1:0.4f})")
                best_f1 = epoch_f1
                best_epoch = epoch
                run.summary["Best F-measure"] = best_f1
                run.summary["Best Epoch"]   = best_epoch
                #best_model_wts = copy.deepcopy(model.state_dict())
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:.4f}'.format(best_f1))
    
    #model.load_state_dict(best_model_wts)

    return model

if __name__ == '__main__':
    main()