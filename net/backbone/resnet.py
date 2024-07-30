import torch
import torch.nn as nn
import torchvision 


class ResNet(nn.Module):
    def __init__(self, args): 
        super().__init__() 
        if 'resnet50' == args.backbone_name:
            self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        elif 'resnet101' == args.backbone_name:
            self.model = torchvision.models.resnet101(weights='IMAGENET1K_V1')

        elif 'resnext50_32x4d' == args.backbone_name:
            self.model = torchvision.models.resnext50_32x4d(weights='IMAGENET1K_V1')
        elif 'resnext101_32x8d' == args.backbone_name:
            self.model = torchvision.models.resnext101_32x8d(weights='IMAGENET1K_V1')
        elif 'resnext101_64x4d' == args.backbone_name:
            self.model = torchvision.models.resnext101_64x4d(weights='IMAGENET1K_V1')

        else:
            raise NotImplementedError('The Resnet backbone architecture not recognized!') 
        
        self.model.avgpool = None
        self.model.fc = None
 

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) 
        x = self.model.layer1(x)
        x = self.model.layer2(x)  
        x = self.model.layer3(x)  
        x = self.model.layer4(x)  
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        return x
    


if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "resnext50_32x4d"
    m = ResNet(args=a())
    y = m(x)
    print(y.shape)

 
 