import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from timm.models.vision_transformer import Mlp

from .gem_head import GeMHead
from .gem_head import GeneralizedMeanPooling
 
 
class TokenMudule(nn.Module):
    def __init__(self, kernel_size=3, dim=768, mlp_ratio=1, nc=4096):
        super().__init__()  
        self.token_block = TokenBlock(kernel_size=kernel_size, dim=dim, mlp_ratio=mlp_ratio)  
        self.head = GeMHead(w_in=dim, nc=nc)
    
    def forward(self, patch_tokens):
        tokens = self.token_block(patch_tokens) 
        global_descriptor = self.head(tokens)
        return global_descriptor 


class TokenBlock(nn.Module):
    def __init__(self, kernel_size=3, dim=768, mlp_ratio=1):
        super().__init__()   
        self.space_self_aware = SpaceSelfAware(kernel_size=kernel_size)
        self.space_fusion =  nn.Sequential(L2Norm(), GeneralizedMeanPooling(norm=3.0))
        self.channel = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio))  
    
    def forward(self, patch_tokens):
        tokens = self.space_self_aware(patch_tokens) 
        tokens = self.space_fusion(tokens).squeeze(-1).squeeze(-1) # b, hw, c, 3, 3 —> b, hw, c  

        tokens = tokens + patch_tokens 
        tokens = nn.functional.normalize(tokens, p=2, dim=-1)  

        tokens = self.channel(tokens)
        return tokens 
    

class SpaceSelfAware(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) 
        self.pad = nn.ZeroPad2d((math.ceil((kernel_size-1)/2), math.floor((kernel_size-1)/2), (kernel_size-1), 0)) # 左右上下
        self.unfold = nn.Unfold(kernel_size=self.kernel_size)

    def forward(self, patch_tokens):
        b, hw, c = patch_tokens.shape

        x = patch_tokens.unsqueeze(-1).unsqueeze(-1)

        h = w = int(math.sqrt(hw))
        patch_tokens = patch_tokens.permute(0, 2, 1).view(b, c, h, w) # b, hw, c —> b, c, hw —> b, c, h, w
        # 下面操作总的形状变化为 b, c, h, w —> b, hw, c, 3, 3
        # 下面操作可以拿到卷积视角下的所有滑动窗口。一共 hw 个窗口，单个窗口形状为 [c, 3, 3]
        y = self.unfold(self.pad(patch_tokens))
        y = y.view(b, c, self.kernel_size[0], self.kernel_size[1], -1)
        y = y.permute(0, 4, 1, 2, 3)
 
        space_self_aware_token = x * y #  b, hw, c, 1, 1 * b, hw, c, 3, 3 —> b, hw, c, 3, 3  

        return space_self_aware_token.contiguous() 
  

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim) 
 
 

if __name__ == "__main__":
    x = torch.randn(8, 256, 768)
    m = TokenMudule()
    y = m(x)
    print(y.shape)