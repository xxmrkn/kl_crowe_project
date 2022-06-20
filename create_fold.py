import dataset
from dataset import TestDataset,TrainDataset
from torch.utils.data import Dataset,DataLoader
from utils.Configuration import CFG

def prepare_loaders(fold,data_df):
    train_df = data_df.query("fold != @fold").reset_index(drop=True)
    valid_df = data_df.query("fold == @fold").reset_index(drop=True)
    #print(train_df['ID'],valid_df['ID'])

    train_dataset = TrainDataset(train_df, transform=dataset.get_transforms('train'))
    valid_dataset = TestDataset(valid_df, transform=dataset.get_transforms('valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                                num_workers=4, shuffle=False, pin_memory=True)
    return train_loader,valid_loader