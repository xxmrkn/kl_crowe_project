import matplotlib.pyplot as plt
import numpy as np
from utils.Configuration import CFG

def visualize_confusion_matrix(matrix, rowlabels, columnlabels):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
    plt.xlabel('pred')
    plt.ylabel('true')

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(rowlabels, minor=False)
    ax.set_yticklabels(columnlabels, minor=False)
    #fig.colorbar(im,ax=ax)
    plt.savefig("/win/salmon/user/masuda/project/vit_kl_crowe/fig/confusion.png")

def visualize_plot_loss(ptl,pvl):
    #loss
    plt.title("Loss",fontsize=18)
    plt.xlabel("Epoch",fontsize=14)
    plt.ylabel("Loss",fontsize=14)

    plt.ylim(0, 3.0)
    plt.xlim(0, CFG.epochs+1, 10)

    plt.xticks(np.arange(0, CFG.epochs+1, 10))
    plt.plot(range(1, CFG.epochs+1),ptl,label='Training Loss',marker ='o')
    plt.plot(range(1, CFG.epochs+1),pvl,label='Validation Loss',marker ='o')
    plt.legend(frameon=False, fontsize=14)

    #plt.show()
    plt.savefig("/win/salmon/user/masuda/project/vit_kl_crowe/fig/loss30.png")
    plt.clf()

def visualize_plot_acc(pta,pva):
    #acc
    plt.title("Accuracy",fontsize=18)
    plt.xlabel("Epoch",fontsize=14)
    plt.ylabel("Accuracy",fontsize=14)

    plt.ylim(0.0, 1.0)
    plt.xlim(0, CFG.epochs+1, 10)

    plt.xticks(np.arange(0, CFG.epochs+1, 10))
    plt.plot(range(1,CFG.epochs+1),pta,label='Training Accuracy',marker ='o')
    plt.plot(range(1,CFG.epochs+1),pva,label='Validation Accuracy',marker ='o')
    plt.legend(frameon=False, fontsize=14)

    #plt.show()
    plt.savefig("/win/salmon/user/masuda/project/vit_kl_crowe/fig/acc30.png")
    plt.clf()

def visualize_plot_f1(ptf,pvf):
    #f1
    plt.title("F1-Score",fontsize=18)
    plt.xlabel("Epoch",fontsize=14)
    plt.ylabel("F1-Score",fontsize=14)

    plt.ylim(0.0, 1.0)
    plt.xlim(0, CFG.epochs+1, 10)

    plt.xticks(np.arange(0, CFG.epochs+1, 10))
    plt.plot(range(1, CFG.epochs+1),ptf,label='Training F1',marker ='o')
    plt.plot(range(1, CFG.epochs+1),pvf,label='Validation F1',marker ='o')
    plt.legend(frameon=False, fontsize=14)

    #plt.show()
    plt.savefig("/win/salmon/user/masuda/project/vit_kl_crowe/fig/f130.png")
    plt.clf()