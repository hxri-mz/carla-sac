import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class ImitationDataloader(Dataset):
    def __init__(self, sensor_root, nav_root, device, transform=None):
        self.sensor_root = sensor_root
        self.nav_root = nav_root
        self.transform = transform
        self.device = device
        
        self.sensor_list = []
        self.nav_list = []
        
        for scene in os.listdir(self.sensor_root):
            scene_sensor_dir = os.path.join(self.sensor_root, scene)
            scene_nav_dir = os.path.join(self.nav_root, scene)
            
            if os.path.isdir(scene_sensor_dir) and os.path.isdir(scene_nav_dir):
                scene_sensor_files = [os.path.join(scene, f) for f in os.listdir(scene_sensor_dir) if f.endswith('.png')]
                scene_nav_files = [os.path.join(scene, f) for f in os.listdir(scene_nav_dir) if f.endswith('.npy')]
                
                self.sensor_list.extend(scene_sensor_files)
                self.nav_list.extend(scene_nav_files)
        
        # self.sensor_list.sort()
        # self.nav_list.sort()
    
    def __len__(self):
        return len(self.sensor_list)
    
    def __getitem__(self, idx):
        filename = list(self.nav_list)[idx].split('/')[-1].split('.')[0]
        scene = list(self.nav_list)[idx].split('/')[0]
        
        if idx == 0:
            prev_file = filename
        else:
            prev_file = list(self.nav_list)[idx-1].split('/')[-1].split('.')[0]
        
        if idx < len(self.sensor_list) - 1:
            next_file = list(self.nav_list)[idx+1].split('/')[-1].split('.')[0]
        else:
            next_file = filename
        
        sensor_path = os.path.join(self.sensor_root, scene, filename + '.png')
        sensor = Image.open(sensor_path).convert('RGB')

        nav_path = os.path.join(self.nav_root, scene, filename + '.npy')
        nav = np.load(nav_path, allow_pickle=True)
        nav = np.concatenate((nav[:-1], nav[-1]), axis=-1).astype(np.float32)
        
        nav_path_prev = os.path.join(self.nav_root, scene, prev_file + '.npy')
        nav_prev = np.load(nav_path_prev, allow_pickle=True)
        nav_prev = np.concatenate((nav_prev[:-1], nav_prev[-1]), axis=-1).astype(np.float32)

        nav_path_next = os.path.join(self.nav_root, scene, next_file + '.npy')
        nav_next = np.load(nav_path_next, allow_pickle=True)
        nav_next = np.concatenate((nav_next[:-1], nav_next[-1]), axis=-1).astype(np.float32)
        
        if self.transform:
            sensor = self.transform(sensor)
        
        return sensor.to(self.device), torch.from_numpy(nav).to(self.device), torch.from_numpy(nav_prev).to(self.device), torch.from_numpy(nav_next).to(self.device)


# if __name__=="__main__":

#     nav_dir = '/mnt/disks/data/carla-sac/dataset/nav/'
#     sensor_dir = '/mnt/disks/data/carla-sac/dataset/sensor/'

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     dataset = ImitationDataloader(sensor_dir, nav_dir, transform=transform, device=device)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

#     for sensor, nav, prev_nav, next_nav in dataloader:
#         print(sensor.shape)
#         print(nav.shape)
