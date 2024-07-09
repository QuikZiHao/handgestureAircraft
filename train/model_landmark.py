import torch
import torch.nn as nn
import numpy as np


class LandMarkModel(nn.Module):
    def __init__(self, input_size:int, output_size:int):
        super(LandMarkModel, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features= output_size, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x) -> torch.Tensor:
        return self.sigmoid(self.fc_layer(x))
    
    def get_score(self, preprocess_landmark:list[np.ndarray]) ->bool:
        gesture = self.forward(torch.tensor(preprocess_landmark))
        if gesture[0] > 0.5:
            return True
        return False