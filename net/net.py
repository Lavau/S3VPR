import torch
import torch.nn as nn 
import math
from net.helper import get_backbone, get_aggregator


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args 
        self.backbone = get_backbone(args)  
        self.aggregator = get_aggregator(args) 

    def forward(self, x): 
        feature = self.backbone(x) 
        if 'mixvpr' == self.args.aggregator_name or\
            'ssm' == self.args.aggregator_name: 
            b,hw,c = feature.shape
            h = w = int(math.sqrt(hw))
            feature = feature.permute(0,2,1).view(b, c, h, w)
        global_descriptor = self.aggregator(feature)  
        return global_descriptor  
        

    # def forward_dinov2_token_module(self, x):
    #     feature = self.backbone(x) 
    #     global_descriptor = self.aggregator(feature)  
    #     return global_descriptor
    


