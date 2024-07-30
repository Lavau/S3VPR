import torch
import torch.nn as nn 


class Dinov2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = torch.hub.load('dinov2', "dinov2_vitb14", source='local') 
        self.num_trainable_blocks = args.num_trainable_blocks  

    def forward(self, x):
        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()
 
        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]: 
            x = blk(x)  

        return x[:, 1:] 

 