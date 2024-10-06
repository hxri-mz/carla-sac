import torchvision.transforms as transforms
from PIL import Image
import torch
from encoder.zoo import EncoderZoo
from parameters import *
import numpy as np
from sim.connection import carla


class TransformObservation():
    def __init__(self, device) -> None:
        self.device = device
        self.encoder_zoo = EncoderZoo(CODE_SIZE)
        self.encoder = self.encoder_zoo.get_model('resnet50')
        self.encoder.to(device)
        self.encoder.eval()

    def preprocess_obs(self, obs):
        obs = Image.fromarray(obs, 'RGB') 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image_tensor = transform(obs)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    # def transform(self, obs):
    #     nav_obs = torch.tensor(obs[1], dtype=torch.float).to(self.device)
    #     im = obs[0]
    #     im.convert(carla.ColorConverter.CityScapesPalette)
    #     placeholder = np.frombuffer(im.raw_data, dtype=np.dtype("uint8"))
    #     placeholder1 = placeholder.reshape((im.height, im.width, 4))
    #     target = placeholder1[:, :, :3]
    #     target = target[:, :, ::-1]
        
    #     im_obs = self.preprocess_obs(target)
        
    #     with torch.no_grad():
    #         im_obs = self.encoder(im_obs)
    #     im_obs = im_obs.to(self.device)
    #     cobs = torch.cat((im_obs, nav_obs.view(1, nav_obs.shape[0])), -1)
    #     return cobs
    
    def transform(self, obs):
        # im_obs = self.preprocess_obs(obs)
        with torch.no_grad():
            im_obs = self.encoder(obs)
        im_obs = im_obs.to(self.device)
        return im_obs
        
