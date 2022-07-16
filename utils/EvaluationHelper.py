import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix

from utils.Configuration import CFG

class EvaluationHelper:

    def f_measure(y_pred, y_true):
        return f1_score(y_true,y_pred,average="macro")
    
    def accuracy(y_pred, y_true):
        return accuracy_score(y_true,y_pred)
    
    def conf_mtrx(y_pred, y_true):
        return confusion_matrix(y_true,y_pred,labels=CFG.labels)

    def one_mistake_acc(matrix,dataset_size):
        taikaku1 = sum(np.diag(matrix)) #対角成分
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1)) #対角成分の両サイド
        other1 = dataset_size-taikaku1#normal
        other2 = dataset_size-taikaku1-taikaku2 #1 neighbor
        # print(taikaku1,taikaku2)
        print(f'taikaku1:{taikaku1},taikaku2:{taikaku2},taikaku_total:{taikaku1+taikaku2}')
        print(f'dataset_size:{dataset_size},other1:{other1},other2:{other2}')
        return (taikaku1+taikaku2)/dataset_size,other1,other2

    def total_acc(matrix,dataset_size):
        taikaku1 = sum(np.diag(matrix)) #対角成分
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1)) #対角成分の両サイド
        other1 = dataset_size-taikaku1#normal
        other2 = dataset_size-taikaku1-taikaku2 #1 neighbor

        print(f'taikaku1:{taikaku1},taikaku2:{taikaku2}')
        print(f'dataset_size:{dataset_size},other1:{other1},other2:{other2}')

        return taikaku1/dataset_size,(taikaku1+taikaku2)/dataset_size,other1,other2

    def total_report(y_pred, y_true):
        return classification_report(y_true,y_pred)

    def criterion(y_pred, y_true):
        return nn.CrossEntropyLoss(y_pred,y_true)

    def index_multi(pred_list, num):
        return [i for i, _num in enumerate(pred_list) if _num == num]
