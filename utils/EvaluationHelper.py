import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix

from utils.Configuration import CFG

class EvaluationHelper:

    def f_measure(y_pred, y_true):
        return f1_score(y_pred,y_true,average="macro")
    
    def accuracy(y_pred, y_true):
        return accuracy_score(y_pred,y_true)
    
    def conf_mtrx(y_pred, y_true):
        return confusion_matrix(y_true,y_pred,labels=CFG.labels)

    def criterion(y_pred, y_true):
        return nn.CrossEntropyLoss(y_pred,y_true)