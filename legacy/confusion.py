from sklearn.metrics import confusion_matrix
import numpy as np

def main():
    y_true = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]

    cm = confusion_matrix(y_true, y_pred)

    print(cm)
    tensor =  np.array([[5,1],[2,7]])
    print(tensor)
    print(cm+tensor)

    print(type(cm))

    return cm

    def one_mistake_acc(matrix):
        taikaku1 = sum(np.diag(matrix)) #対角成分
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1)) #対角成分の両サイド
        return taikaku1+taikaku2

if __name__ == '__main__':
    main()