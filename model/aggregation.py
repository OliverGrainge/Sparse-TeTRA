
import torch 
import torch.nn as nn 

class CLS(nn.Module): 
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x[:, 0, :]