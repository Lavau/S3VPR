import torch  
from torchvision.models.vision_transformer import vit_b_16, vit_b_32


class ViT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.backbone_name == 'vit_b_16': 
            self.model = vit_b_16(weights="IMAGENET1K_V1")
        elif args.backbone_name == 'vit_b_32': 
            self.model = vit_b_32(weights="IMAGENET1K_V1") 
        else:
            raise NotImplementedError('The Vit backbone architecture not recognized!')
        self.num_trainable_blocks = args.num_trainable_blocks  

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.encoder.layers[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()
        # Last blocks are trained
        for blk in self.model.encoder.layers[-self.num_trainable_blocks:]: 
            x = blk(x) 
       
        return x[:,1:,:]



if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "vit_b_32"
    m = ViT(args=a())
    y = m(x)
    print(y.shape)










# import torch 
# from timm.models.vision_transformer import vit_base_patch8_224, vit_base_patch16_224, vit_base_patch32_224 


# class ViT(torch.nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         if args.backbone_name == 'vit_base_patch8': 
#             self.model = vit_base_patch8_224()
#         elif args.backbone_name == 'vit_base_patch16': 
#             self.model = vit_base_patch16_224()
#         elif args.backbone_name == 'vit_base_patch32': 
#             self.model = vit_base_patch32_224()
#         else:
#             raise NotImplementedError('The Vit backbone architecture not recognized!')
#         self.num_trainable_blocks = args.num_trainable_blocks  

#     def forward(self, x):
#         x = self.model.patch_embed(x)
#         x = self.model._pos_embed(x)
#         x = self.model.norm_pre(x) 
#         # First blocks are frozen
#         with torch.no_grad():
#             for blk in self.model.blocks[:-self.num_trainable_blocks]:
#                 x = blk(x)
#         x = x.detach() 
#         # Last blocks are trained
#         for blk in self.model.blocks[-self.num_trainable_blocks:]: 
#             x = blk(x)
#         return x[:, 1:, :] 



# if __name__ == "__main__":
#     x = torch.randn(1,3,224,224) 
#     class a:
#         num_trainable_blocks = 4
#         backbone_name = "vit_base_patch32"
#     m = ViT(args=a())
#     y = m(x)
#     print(y.shape)


