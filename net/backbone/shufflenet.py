import torch
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0, shufflenet_v2_x0_5,\
      shufflenet_v2_x1_5, shufflenet_v2_x2_0


class ShuffleNet(torch.nn.Module):
    def __init__(self, args):  
        super().__init__() 
        if 'shufflenet_v2_x1_0' == args.backbone_name:
            self.model = shufflenet_v2_x1_0(weights='IMAGENET1K_V1') 
        elif 'shufflenet_v2_x0_5' == args.backbone_name:
            self.model = shufflenet_v2_x0_5(weights='IMAGENET1K_V1') 
        elif 'shufflenet_v2_x1_5' == args.backbone_name:
            self.model = shufflenet_v2_x1_5(weights='IMAGENET1K_V1') 
        elif 'shufflenet_v2_x2_0' == args.backbone_name:
            self.model = shufflenet_v2_x2_0(weights='IMAGENET1K_V1') 
        else:
            raise NotImplementedError(f'The ShuffleNet {args.backbone_name} backbone architecture not recognized!') 
        self.model.fc = None
         
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        b,c,h,w = x.shape
        x = x.view(b,c,h*w).permute(0,2,1)
        return x


if __name__ == "__main__":
    x = torch.randn(1,3,224,224) 
    class a:
        num_trainable_blocks = 4
        backbone_name = "shufflenet_v2_x1_5"
    m = ShuffleNet(args=a())
    for name, param in m.named_parameters():
        print(f"{name}")
    y = m(x)
    print(y.shape) #torch.Size([1, 49, 1024])

   