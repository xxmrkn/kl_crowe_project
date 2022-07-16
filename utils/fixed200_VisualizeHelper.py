import cv2
import csv
import math
import pickle
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
        if num>=200:
            plt.subplots_adjust(hspace=0.5)
        #plt.subplots_adjust(wspace=0.7)
        plt.axis("off")

    if flag==1:
        plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
        file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/outlier/fixed200_{CFG.model_name}_fold{fold}_epoch{epoch}{CFG.epochs}_outliers.pdf"
        plt.savefig(file_name)
        print('--> Saved Outlier images (nomal accuracy)')
        plt.close()
    elif flag==2:
        plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
        file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/outlier2/fixed200_{CFG.model_name}_fold{fold}_epoch{epoch}{CFG.epochs}_outliers2.pdf"
        plt.savefig(file_name)
        print('--> Saved Outlier images (1 neighbor accuracy)')
        plt.close()

#last epoch and fold
def visualize_total_image(path,id,labels1,labels2,num,normal_acc,neighbor_acc,flag):
    plt.figure(figsize=(150,150))

    for x,(image_path,im_id,label1,label2) in enumerate(zip(path,id,labels1,labels2)):
        plt.subplot(math.ceil(num/16),16,x+1)
        #print(image_path,id,label1,label2)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"{im_id[:-4]}\nactual class : {label1}\npred class : {label2}\n", fontsize=30)
        if num>=350:
            plt.subplots_adjust(hspace=1.2)
        elif num>=200:
            plt.subplots_adjust(hspace=0.5)
        
        #plt.subplots_adjust(wspace=0.7)
        plt.axis("off")
    if flag==1:
        new = [[0]*3 for _ in range(len(id))]
        #print(len(id),len(labels1),len(labels2))

        for i in range(len(id)):
            new[i][0],new[i][1],new[i][2] = id[i],labels1[i],labels2[i]

        #print(new)

        with open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/fixed200_csv_to_pptx_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outlier.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new)
        print('--> Saved Total Outlier csv')

        plt.suptitle(f'Model:{CFG.model_name}  Epoch:{CFG.epochs}  Fold:{CFG.n_fold}  \nNumber of images:{num}/400  Normal Accuracy:{normal_acc:.4f}',fontsize=100)
        file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/fixed200_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outliers.pdf"
        plt.savefig(file_name)
        print('--> Saved Total Outlier images (normal accuracy)')
        plt.close()
    else:
        new2 = [[0]*3 for _ in range(len(id))]

        for i in range(len(path)):
            new2[i][0],new2[i][1],new2[i][2] = id[i],labels1[i],labels2[i]

        #print(new2)

        with open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/fixed200_csv_to_pptx_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outlier2.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new2)
        print('--> Saved Total Outlier2 csv')

        plt.suptitle(f'Model:{CFG.model_name}  Epoch:{CFG.epochs}  Fold:{CFG.n_fold}  \nNumber of images:{num}/400  1Neighbor Accuracy:{neighbor_acc:.4f}',fontsize=100)
        file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/fixed200_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outliers2.pdf"
        plt.savefig(file_name)
        print('--> Saved Total Outlier images (1 neighbor accuracy)')
        plt.close()



# import cv2
# import math
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from utils.Configuration import CFG

# def visualize_confusion_matrix(matrix, rowlabels, columnlabels):

#     fig, ax = plt.subplots()
#     heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)

#     ax.set_xticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
#     ax.set_yticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
#     plt.xlabel('pred')
#     plt.ylabel('true')

#     ax.invert_yaxis()
#     ax.xaxis.tick_top()

#     ax.set_xticklabels(rowlabels, minor=False)
#     ax.set_yticklabels(columnlabels, minor=False)
#     #fig.colorbar(im,ax=ax)
#     plt.savefig("fig/confusion.png")

# def visualize_image(path,id,labels1,labels2,num,flag,fold,epoch):
#     plt.figure(figsize=(150,150))

#     for x,(image_path,id,label1,label2) in enumerate(zip(path,id,labels1,labels2)):
#         plt.subplot(math.ceil(num/16),16,x+1)
#         #print(image_path,id,label1,label2)

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
#         plt.imshow(image)
#         plt.title(f"{id[:-4]}\nactual class : {label1}\npred class : {label2}\n", fontsize=30)
#         if num>=200:
#             plt.subplots_adjust(hspace=0.5)
#         #plt.subplots_adjust(wspace=0.7)
#         plt.axis("off")

#     if flag==1:
#         plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
#         file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/outlier/{CFG.model_name}_fold{fold}_epoch{epoch}{CFG.epochs}_outliers.pdf"
#         plt.savefig(file_name)
#         print('--> Saved Outlier images (nomal accuracy)')
#         plt.close()
#     elif flag==2:
#         plt.suptitle(f'MODEL:{CFG.model_name} EPOCH:{epoch}/{CFG.epochs} FOLD:{fold}/{CFG.n_fold}',fontsize=80)
#         file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/outlier2/{CFG.model_name}_fold{fold}_epoch{epoch}{CFG.epochs}_outliers2.pdf"
#         plt.savefig(file_name)
#         print('--> Saved Outlier images (1 neighbor accuracy)')
#         plt.close()

# #last epoch and fold
# def visualize_total_image(path,id,labels1,labels2,num,normal_acc,neighbor_acc,flag):
#     plt.figure(figsize=(150,150))

#     for x,(image_path,id,label1,label2) in enumerate(zip(path,id,labels1,labels2)):
#         plt.subplot(math.ceil(num/16),16,x+1)
#         #print(image_path,id,label1,label2)

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
#         plt.imshow(image)
#         plt.title(f"{id[:-4]}\nactual class : {label1}\npred class : {label2}\n", fontsize=30)
#         if num>=350:
#             plt.subplots_adjust(hspace=1.2)
#         elif num>=200:
#             plt.subplots_adjust(hspace=0.5)
        
#         #plt.subplots_adjust(wspace=0.7)
#         plt.axis("off")
#     if flag==1:
#         f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/path_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outlier.txt","wb")
#         list = [path,labels1,labels2]
#         pickle.dump(list,f)
#         print('--> Saved Outlier path')

#         plt.suptitle(f'Model:{CFG.model_name}  Epoch:{CFG.epochs}  Fold:{CFG.n_fold}  \nNumber of images:{num}/944  Normal Accuracy:{normal_acc:.4f}',fontsize=100)
#         file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outliers.pdf"
#         plt.savefig(file_name)
#         print('--> Saved Total Outlier images (normal accuracy)')
#         plt.close()
#     else:
#         f = open(f"outputs/{CFG.model_name}/{CFG.n_fold}fold/path_{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outlier2.txt","wb")
#         list = [path,labels1,labels2]
#         pickle.dump(list,f)
#         print('--> Saved Outlier2 path')

#         plt.suptitle(f'Model:{CFG.model_name}  Epoch:{CFG.epochs}  Fold:{CFG.n_fold}  \nNumber of images:{num}/944  1Neighbor Accuracy:{neighbor_acc:.4f}',fontsize=100)
#         file_name = f"fig/{CFG.model_name}/{CFG.n_fold}fold/{CFG.model_name}_{CFG.n_fold}fold_{CFG.epochs}epoch_outliers2.pdf"
#         plt.savefig(file_name)
#         print('--> Saved Total Outlier images (1 neighbor accuracy)')
#         plt.close()