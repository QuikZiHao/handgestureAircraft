import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandMarkDataSet(Dataset):
    def __init__(self, root_dir:str):
        self.root_dir = root_dir
        self.data = []
        self.label = []
        self.type_amt = 0

        # Collect all csv file paths
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path) and folder.isnumeric():
                self.type_amt += 1
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(folder_path, file_name)
                        # Load the data from the CSV file
                        df = pd.read_csv(file_path)
                        # Convert the data to a tensor
                        tensor_data = torch.tensor(df.values, dtype=torch.float32)
                        for i in range(len(tensor_data)):
                            self.data.append(tensor_data[i])
                            self.label.append(int(folder))
          
    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor_data, label = self.data[idx], self.label[idx]
        tensor_label = torch.tensor(label, dtype=torch.long)
        
        return tensor_data, tensor_label
    
    def get_type_amt(self) ->int :
        return self.type_amt
    