import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
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
    plt.savefig("fig/confusion.png")

def visualize_image(path,id,labels1,labels2,num,flag,fold,epoch):
    plt.figure(figsize=(150,150))

    for x,(image_path,id,label1,label2) in enumerate(zip(path,id,labels1,labels2)):
        plt.subplot(math.ceil(num/16),16,x+1)
        #print(image_path,id,label1,label2)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"{id[:-4]}\nactual class : {label1}\npred class : {label2}\n", fontsize=30)
        #plt.subplots_adjust(wspace=0.7)
        plt.axis("off")

    if flag==1:
        plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
        file_name = f"fig/{CFG.model_name}/{CFG.model_name}_fold{fold}_epoch{epoch}_outliers.pdf"
        plt.savefig(file_name)
        print('--> Saved Outlier images (nomal accuracy)')
        plt.close()
    else:
        plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
        file_name = f"fig/{CFG.model_name}/{CFG.model_name}_fold{fold}_epoch{epoch}_outliers2.pdf"
        plt.savefig(file_name)
        print('--> Saved Outlier images (1 neighbor accuracy)')
        plt.close()