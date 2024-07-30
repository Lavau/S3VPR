import torch 
from timm.models.swin_transformer import swin_base_patch4_window7_224, \
    swin_base_patch4_window7_224_in22k, swin_s3_base_224
from timm.models.swin_transformer_v2_cr import swinv2_cr_base_224, swinv2_cr_base_ns_224


class Swin(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.backbone_name == 'swin_b_p4_w7': 
            self.model = swin_base_patch4_window7_224(pretrained=True) 
        elif args.backbone_name == 'swin_b_p4_w7_in22k': 
            self.model = swin_base_patch4_window7_224_in22k(pretrained=True)
        elif args.backbone_name == 'swin_s3_b': 
            self.model = swin_s3_base_224(pretrained=True)
        elif args.backbone_name == 'swinv2_cr_b': 
            self.model = swinv2_cr_base_224(pretrained=True)
        elif args.backbone_name == 'swinv2_cr_b_ns': 
            self.model = swinv2_cr_base_ns_224(pretrained=True)
        else:
            raise NotImplementedError('The Swin backbone architecture not recognized!')
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