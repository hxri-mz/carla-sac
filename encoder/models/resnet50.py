import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Encoder(nn.Module):
    def __init__(self, code_size=94):
        super(ResNet50Encoder, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=True)        
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        for param in self.resnet50.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet50(x)
        x = nn.functional.relu(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return x