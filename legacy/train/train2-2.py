#kaggleの優秀なコード書き換えてたやつ
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

#from IPython.display import display
#from IPython import display as ipd

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

import wandb

import dataset


def main():

    #Define model
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
    
    #Set seed
    CFG.set_seed(CFG.seed)

    #Prepare Dataframe
    data_df = pd.read_csv(CFG.csv_path)
    data_df["target"] = data_df["Crowe"]+data_df["KL"]

    #Training and Validation
    train_df,valid_df = train_test_split(data_df,test_size=0.3)

    #Prepare Dataset
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
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
    num_epochs = CFG.epochs

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
    
    for epoch in range(num_epochs):
        gc.collect()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        scaler = amp.GradScaler()
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_acc = 0.0
            running_f1 = 0.0
            
            pbar = tqdm(enumerate(dataloaders[phase],start=1), total=len(dataloaders[phase]))
            for step, (inputs,labels) in pbar:
                #print(step,phase)
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                if phase=='train':
                    with amp.autocast(enabled=True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)

                        acc = EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
                        f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
                        loss = criterion(outputs,labels)

                        running_loss += loss.item()
                        running_acc += acc
                        running_f1 += f_m

                        
                        train_loss = running_loss / step
                        train_acc = running_acc / step
                        train_f1 = running_f1 / step

                    scaler.scale(loss).backward()

                    if (step) % CFG.n_accumulate==0:
                        scaler.step(optimizer)
                        scaler.update()

                        #zero the parameter gradients
                        optimizer.zero_grad()

                        if scheduler is not None:
                            scheduler.step()

                if phase=='valid':
                    #with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)

                        acc = EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
                        f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
                        loss = criterion(outputs,labels)

                        running_loss += loss.item()
                        running_acc += acc
                        running_f1 += f_m

                        valid_loss = running_loss / step
                        valid_acc = running_acc / step
                        valid_f1 = running_f1 / step
        if phase=='train':
            history['Train Loss'].append(train_loss)
            history['Train Accuracy'].append(train_acc)
            history['Train F-measure'].append(train_f1)
        else:
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

            print(f"Train Loss: {train_loss} Train Acc: {train_acc} Train f1: {train_f1}")
            print(f"Valid Loss: {valid_loss} Valid Acc: {valid_acc} Valid f1: {valid_f1}")
            
            if phase == 'valid' and valid_f1 > best_f1:
                print(f"Valid Score Improved ({best_f1:0.4f} ---> {valid_f1:0.4f})")
                best_f1 = valid_f1
                best_epoch = epoch
                run.summary["Best F-measure"] = best_f1
                run.summary["Best Epoch"]   = best_epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                #PATH = f"best_epoch-{epoch:02d}.bin"
                #torch.save(model.state_dict(), PATH)
                #wandb.save(PATH)

        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60}m {time_elapsed % 60}s')
    print('Best val F1: {:.4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)

    run.finish()

    #display wandb webpage link
    print(f"wandb website ------> {run.url}")

    #display(ipd.IFrame(run.url, width=1000, height=720))
    #remove wandb files
    shutil.rmtree("wandb")

    return model


if __name__ == '__main__':
    main()