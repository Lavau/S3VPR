import torch
from torchvision.models.efficientnet import efficientnet_b0


class EfficientNet(torch.nn.Module):
    def __init__(self, args):  
        super().__init__() 
        if 'efficientnet' == args.backbone_name:
            self.model = efficientnet_b0(weights='IMAGENET1K_V1') 
        else:
            raise NotImplementedError('The EfficientNet backbone architecture not recognized!') 
        self.model.classifier = None
        self.model.avgpool = None
         
        
    def forward(self, x):
        x = self.model.features(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        return x


if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "efficientnet"
    m = EfficientNet(args=a())
    y = m(x)
    print(y.shape) #torch.Size([1, 49, 1280])

    for name, param in m.named_parameters():
        print(f"{name}")