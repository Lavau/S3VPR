import torch
import torchvision


class MobileNetv2(torch.nn.Module):
    def __init__(self, args): 
        super().__init__() 
        if 'mobilenet_v2' == args.backbone_name:
            self.model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1') 
        else:
            raise NotImplementedError('The Resnet backbone architecture not recognized!') 
        self.model.classifier = None 

    def forward(self, x):
        x = self.model.features(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        return x


if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "mobilenet_v2"
    m = MobileNetv2(args=a())
    y = m(x)
    print(y.shape)