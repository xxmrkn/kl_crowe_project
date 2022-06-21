import gc
from glob import glob
from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from utils.Configuration import CFG
from utils.EvaluationHelper import EvaluationHelper

def train_one_epoch(model, optimizer, scheduler, criterion, dataloader, device):
    model.train()
    grad_scaler = torch.cuda.amp.GradScaler()
    
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    dataset_size = 0
    
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc='Train ')
    for step, (inputs, labels) in pbar:         
        inputs = inputs.to(device)
        labels  = labels.to(device)
        
        batch_size = inputs.size(0)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            acc = EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
            f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
            loss = criterion(outputs,labels)

            running_loss += loss.item()
            running_acc += acc
            running_f1 += f_m

            train_loss = running_loss / step
            train_acc = running_acc / step
            train_f1 = running_f1 / step

            #print(optimizer.param_groups[0]['lr'])

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        #if scheduler is not None:
        #    scheduler.step()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{train_loss:0.6f}',
                        train_acc=f'{train_acc:0.6f}',
                        train_f1=f'{train_f1:0.6f}',
                        lr=f'{current_lr:0.6f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
            
        gc.collect()
    
    return train_loss, train_acc, train_f1

#@torch.no_grad()
def valid_one_epoch(model, optimizer, criterion, dataloader, device, id_list, id_list2, tmp_acc):
    with torch.no_grad():
        model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0

        c_mat = 0
        
        pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc='Valid ')
        for step, (inputs, labels, image_path) in pbar:        
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            #optimizer.zero_grad()

            batch_size = inputs.size(0)

            outputs  = model(inputs)
                
            _, preds = torch.max(outputs, 1)

            #print(labels,preds)

            acc = EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
            f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
            cm  = EvaluationHelper.conf_mtrx(CFG.to_numpy(preds), CFG.to_numpy(labels))
            #print(cm.shape)
            loss = criterion(outputs,labels)

            running_loss += loss.item()
            running_acc += acc
            running_f1 += f_m
            dataset_size += batch_size
            c_mat += cm

            valid_loss = running_loss / step
            valid_acc = running_acc / step
            valid_f1 = running_f1 / step

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{valid_loss:0.6f}',
                            valid_acc=f'{valid_acc:0.6f}',
                            valid_f1=f'{valid_f1:0.6f}',
                            lr=f'{current_lr:0.6f}',
                            gpu_memory=f'{mem:0.2f} GB')
            torch.cuda.empty_cache()

            #log miss classes
            # if tmp_acc < valid_acc:
            #     tmp_acc = acc
            #     diff = abs(preds - labels)
            #     for i in range(2,CFG.num_classes):
            #         tmp = EvaluationHelper.index_multi(diff, i)
            #         #print(tmp)
            #         for j in range(len(tmp)):
            #             id_list[i-2].append(image_path[tmp[j]])
            #print(id_list)

            #log miss classes
            diff = preds - labels #calculate diff
            #print(f'preds:{preds},labels:{labels}')
            #print(diff)
            #print(-CFG.num_classes)
            
            #for visualize outliers
            for i in range(-CFG.num_classes+1,CFG.num_classes):
                if i in [0]: #remove nomal
                    pass
                else:
                    tmp = EvaluationHelper.index_multi(diff, i)
                    #print(len(tmp))
                    #print(i,tmp)
                    for j in range(len(tmp)):
                        id_list[i+CFG.num_classes-1].append(image_path[tmp[j]])

            for i in range(-CFG.num_classes+1,CFG.num_classes):
                if i in [-1,0,1]: #remove nomal, 1 neighbor
                    pass
                else:
                    tmp = EvaluationHelper.index_multi(diff, i)
                    #print(len(tmp))
                    #print(i,tmp)
                    for j in range(len(tmp)):
                        id_list2[i+CFG.num_classes-1].append(image_path[tmp[j]])
            #print(id_list)

            gc.collect()
    
    return valid_loss, valid_acc, valid_f1, c_mat, dataset_size, id_list, id_list2, tmp_acc
    #return valid_loss, valid_acc, valid_f1, c_mat