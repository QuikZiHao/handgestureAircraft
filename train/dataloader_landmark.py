import torch
from torch.utils.data import DataLoader, random_split

class LandMarkDataLoader:
    def __init__(self,dataset:list[torch.Tensor], test_train_ratio:float=0.8 , batch_size:int=32, shuffle:bool=True):
        self.batch_size = batch_size
        self.test_train_ratio = test_train_ratio
        # Spliting dataset
        self.train_dataset,self.test_dataset =  random_split(dataset, [int(len(dataset)*self.test_train_ratio), len(dataset)-int(len(dataset)*self.test_train_ratio)])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle,drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle,drop_last=False)



