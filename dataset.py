#Import Libraries
import os
import cv2

from utils.Configuration import CFG

from torch.utils.data import Dataset

import albumentations as A
from albumentations import (
    Compose, OneOf, Normalize, CenterCrop, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, RandomRotate90, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout,GridDropout
    )
from albumentations.pytorch import ToTensorV2

#DataAugmentation
def get_transforms(data):
    
    if data == 'train':
        return Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            #A.Rotate(limit=45, p=0.5),
            #A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.5),
            #A.GridDropout(ratio=0.3, unit_size_min=None, unit_size_max=None, holes_number_x=5, holes_number_y=5, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
            #A.ShiftScaleRotate(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.image_size, CFG.image_size),
            Normalize(),
            ToTensorV2(),
        ])

#Dataset
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_ids = df["ID"].values
        self.labels = df["target"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(CFG.image_path,self.df["ID"].iloc[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_ids = df["ID"].values
        self.labels = df["target"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(CFG.image_path,self.df["ID"].iloc[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label