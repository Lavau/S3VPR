import torch

from net.backbone.dinov2 import Dinov2 
from net.backbone.vit import ViT
from net.backbone.deit import DeiT
from net.backbone.resnet import ResNet 
from net.backbone.mobilenetv2 import MobileNetv2
from net.backbone.shufflenet import ShuffleNet
from net.backbone.efficientnet import EfficientNet

from net.aggregator.gem_head import GeMHead 
from net.aggregator.token_module import TokenMudule 
from net.aggregator.mixvpr import MixVPR 


def get_backbone(args):
    if 'dinov2' == args.backbone_name: 
        backbone = Dinov2(args)  
        state_dict = torch.load(args.foundation_model_path) 
        sd = {} 
        for k1, k2 in zip(backbone.state_dict().keys(), state_dict.keys()): 
            sd[k1] = state_dict[k2]   
        backbone.load_state_dict(sd, strict=False)
        return backbone     
    
    elif 'vit' in args.backbone_name: 
        backbone = ViT(args=args) 
        return backbone  

    elif 'deit' in args.backbone_name: 
        backbone = DeiT(args=args) 
        return backbone    

    # CNN
    elif 'resne' in args.backbone_name: 
        backbone = ResNet(args=args) 
        return backbone     
    elif 'mobilenet_v2' == args.backbone_name: 
        backbone = MobileNetv2(args=args) 
        return backbone   
    elif 'efficientnet' == args.backbone_name: 
        backbone = EfficientNet(args=args) 
        return backbone   
    elif 'shufflenet' in args.backbone_name: 
        backbone = ShuffleNet(args=args) 
        return backbone     
    
    else:
        raise NotImplementedError(f'Backbone {args.backbone_name} architecture not recognized!')
    

# resnet50\101, layer3, dim=1024
# resnet50\101, layer4, dim=2048
def get_aggregator(args): 
    if 'token_module' == args.aggregator_name: 
        return TokenMudule(kernel_size=args.kernel_size, dim=args.dim, mlp_ratio=args.mlp_ratio, nc=args.nc) 
    
    elif 'gemhead' == args.aggregator_name: 
        return GeMHead(w_in=args.dim, nc=args.nc) 

    elif 'mixvpr' == args.aggregator_name: 
        return MixVPR(in_channels=768, in_h=16, in_w=16, out_channels=1024, mix_depth=4, mlp_ratio=1, out_rows=4) 
    else:
        raise NotImplementedError('Aggregator architecture not recognized!')
    
 