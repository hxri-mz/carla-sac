import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34Encoder(nn.Module):
    def __init__(self, code_size=94):
        super(ResNet34Encoder, self).__init__()
        
        self.resnet34 = models.resnet34(pretrained=False)
        self.checkpoint = torch.load('encoder/ckpt/resnet34.ckpt')
        self.resnet34.load_state_dict(self.checkpoint['state_dict'])        
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])
        for param in self.resnet34.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet34(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x