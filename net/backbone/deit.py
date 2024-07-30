import torch 
from timm.models.deit import deit_base_patch16_224, deit3_base_patch16_224,\
      deit3_base_patch16_224_in21ft1k


class DeiT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.backbone_name == 'deit_base_patch16_224': 
            self.model = deit_base_patch16_224(pretrained=True) 
        elif args.backbone_name == 'deit3_base_patch16_224': 
            self.model = deit3_base_patch16_224(pretrained=True)
        elif args.backbone_name == 'deit3_base_patch16_224_in21ft1k': 
            self.model = deit3_base_patch16_224_in21ft1k(pretrained=True)
        else:
            raise NotImplementedError('The Vit backbone architecture not recognized!')
        self.num_trainable_blocks = args.num_trainable_blocks  

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x) 
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach() 
        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]: 
            x = blk(x)
        return x[:, 1:, :] 



if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "deit3_base_patch16_224_in21ft1k"
    m = DeiT(args=a())
    y = m(x)
    print(y.shape)

