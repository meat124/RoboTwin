import torch
import torch.nn as nn

from data4robotics.models.base import BaseModel
from DFormer.models.encoders.DFormerv2 import DFormerv2_S, DFormerv2_B


class DFormerv2(nn.Module):
    def __init__(self, model_name="DFormerv2_Small,NYU", restore_path=None):
        super().__init__()
        if model_name in ("DFormerv2_Small_NYU", "DFormerv2_Small_SUNRGBD"):
            model = DFormerv2_S()
        elif model_name in ("DFormerv2_Base_NYU", "DFormerv2_Base_SUNRGBD"):
            model = DFormerv2_B()
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
        # load weights
        if restore_path:
            print("Restoring model from", restore_path)
            model.init_weights(restore_path)
            
        self.model = model
        
    def forward(self, x, x_e):
        B = x.shape[0]
        x = self.model(x, x_e)
        x = x[-1].reshape((B, 512, -1))
        x = torch.mean(x, dim=-1)  # (B, 512)
        x = x.unsqueeze(1)  # (B, 1, 512)
        return x
    
    @property
    def embed_dim(self):
        return 512
    
    @property
    def n_tokens(self):
        return 1