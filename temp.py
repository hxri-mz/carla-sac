import numpy as np
import cv2

# Define the mapping from class id to color
id_to_color = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    # Add more mappings if necessary
}

# Load your semantic segmentation image (size: 1600x900, 3)
# Assuming the image is stored as a numpy array with shape (1600, 900, 3)
segmentation_image = np.load('/mnt/disks/data/carla-sac/dataset/sensor/scene_0/0.npy')  # Example input

# Initialize an empty RGB image
rgb_image = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1], 3), dtype=np.uint8)

# Map each class id to its corresponding color
for id_val, color in id_to_color.items():
    rgb_image[segmentation_image == id_val] = color

# Save or display the resulting image
cv2.imwrite('segmentation_rgb.png', rgb_image)