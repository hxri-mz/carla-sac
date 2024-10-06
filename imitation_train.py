import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from imitation_learning.utils import ImitationDataloader
from imitation_learning.model import ImitationModel

from utils.transform import TransformObservation
from encoder.zoo import EncoderZoo

def penalty(output, nav, input_data):

    # Case 1: [1, 0, 0] should map to a negative value in output[1]
    mask_1 = (nav == torch.tensor([1.0, 0.0, 0.0])).all(dim=1)
    penalty_1 = torch.where(output[mask_1, 1] >= 0, output[mask_1, 1], torch.tensor(0.0)).sum()

    # Case 2: [0, 1, 0] should map to a positive value in output[1]
    mask_2 = (nav == torch.tensor([0.0, 1.0, 0.0])).all(dim=1)
    penalty_2 = torch.where(output[mask_2, 1] <= 0, -output[mask_2, 1], torch.tensor(0.0)).sum()

    # Case 3: [0, 0, 1] should map to a zero value in output[1]
    mask_3 = (nav == torch.tensor([0.0, 0.0, 1.0])).all(dim=1)
    penalty_3 = torch.abs(output[mask_3, 1]).sum()

    # Total penalty term
    total_penalty = penalty_1 + penalty_2 + penalty_3

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = TransformObservation(device)

    obs_dim = 2048 
    code_size = 95
    nav_size = 6
    action_dim = 2 

    model = ImitationModel(obs_dim, code_size, nav_size, action_dim, device)
    print(model)
    criterion = nn.MSELoss()

    sensor_dir = '/mnt/disks/data/carla-sac/dataset/sensor/'
    nav_dir = '/mnt/disks/data/carla-sac/dataset/nav/'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImitationDataloader(sensor_dir, nav_dir, transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    num_epochs = 100
    loss_values = []
    best_loss = float('inf')
    best_model_dir = "imitation_checkpoints" 
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for sensor, nav, prev_nav, next_nav in tqdm(dataloader):
            obs = sensor
            obs = tfm.transform(obs)
            nav_obs = torch.cat((prev_nav[:,3].unsqueeze(-1), nav[:,4].unsqueeze(-1), nav[:, 0].unsqueeze(-1), next_nav[:,5:8]), axis=-1)
            nav_target = nav[:,2:4]
            # import pdb; pdb.set_trace()
            nav_target = torch.flip(nav_target, dims=(1,))

            actions = model(obs, nav_obs)
            
            loss = criterion(actions, nav_target.to(torch.float32))
            
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            
            running_loss += loss.item() * sensor.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        loss_values.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_path = os.path.join(best_model_dir, f"actor_model_{best_loss:.5f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved with loss: {best_loss:.5f}")

    print("Training complete")

    # Plot the loss curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("imitation_loss_curve.png")