import torchvision.transforms as transforms
from PIL import Image
import torch
from encoder.zoo import EncoderZoo
from parameters import *
import numpy as np

class TransformObservation():
    def __init__(self, device) -> None:
        self.device = device
        self.encoder_zoo = EncoderZoo(CODE_SIZE)
        self.encoder = self.encoder_zoo.get_model(ENCODER_MODEL)
        self.encoder.eval()

    # def preprocess_obs(self, obs):
    #     obs = Image.fromarray(obs, 'RGB') 
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ])

    #     image_tensor = transform(obs)
    #     image_tensor = image_tensor.unsqueeze(0)
    #     return image_tensor
    
    def transform(self, obs):
        nav_obs = torch.tensor(obs[1], dtype=torch.float).to(self.device)
        # im_obs = self.preprocess_obs(obs[0])
        im_obs = torch.from_numpy(obs[0] / 255.0).float().unsqueeze(0).permute(0,3,1,2)
        # Image.fromarray((im_obs.permute(0,2,3,1).detach().cpu().numpy()[0, :, :, :]*255.0).astype(np.uint8)).save('tt2.png')
        # import pdb; pdb.set_trace()
        im_obs = self.encoder(im_obs)
        im_obs = im_obs.to(self.device)
        cobs = torch.cat((im_obs, nav_obs.view(1, nav_obs.shape[0])), -1)
        return cobs
    
    # def transform(self, obs):
    #     nav_obs = torch.tensor(obs[1], dtype=torch.float).to(self.device)
    #     im_obs = self.preprocess_obs(obs[0])
    #     im_obs = im_obs.to(self.device)
    #     return im_obs, nav_obs
        
