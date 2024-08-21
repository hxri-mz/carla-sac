import torch
from .models.resnet50 import ResNet50Encoder
from .models.resnet34 import ResNet34Encoder


class EncoderZoo:
    def __init__(self, code_size=95):
        self.models = {
            'resnet50':ResNet50Encoder(code_size),
            'resnet34':ResNet34Encoder(code_size),
        }
        self.code_size = code_size

    def get_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in the zoo.")
        return self.models[name]

    def list_models(self):
        return list(self.models.keys())
