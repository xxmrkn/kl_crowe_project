# kl_crowe_project
##プロジェクト概要
股関節DRRからKL,Croweを分類する．

## ファイルの概要
model.py : VisionTransformer
pretrained_model.py : torchvision VisionTransformer(pretrained)
train.py : for run training
train_skfold.py : train.pyをStratified KFoldで行うようにしたもの
trainval_one_epoch.py : 1epochの訓練と検証

utils.Configuration.py : パラメータなど
utils.EvaluationHelper.py : 評価指標など
utils.VisualizeHelper.py : 可視化のためのコード（wandbを使い始めてから使ってない）

dataset.dataset.py : 訓練とテストのdatasetクラス
dataset.datasetfold.py : foldごとにdatasetクラスを作るため