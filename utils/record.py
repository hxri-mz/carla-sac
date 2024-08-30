import numpy as np
import PIL
import pathlib
from sim.connection import carla
from PIL import Image
SAVE_DIR = '/mnt/disks/data/carla-sac/dataset'

def record_data(sensor_data, nav_data, idx, num, save_dir=SAVE_DIR):
    # image_obs = Image.fromarray(sensor_data)
    nav_obs = np.array(nav_data)
    # Save to directory
    sensor_data.save_to_disk(f'{save_dir}/sensor/scene_{num}/{idx}.png', carla.ColorConverter.CityScapesPalette)
    # np.save(f'{save_dir}/sensor/scene_{num}/{idx}.npy', sensor_data)
    np.save(f'{save_dir}/nav/scene_{num}/{idx}.npy', nav_obs)