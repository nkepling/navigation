from utils import *
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from fo_solver import value_iteration, extract_policy, visualize_policy_and_rewards, visualize_rewards, pick_start_and_goal
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 10  # size of the grid
config = "block"  # distribution of positive probability cells
num_blocks = 3  # number of positive region blocks
num_obstacles = 3  # number of obstacles
obstacle_type = "block"
square_size = 4  # size of the positive region square

# Discount factor
gamma = 0.8

# define experiment configuration
random_map = True


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def reformat_input(rewards, obstacles_map):
    """Reformat the input for the NN model
    """
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
    obstacles_map = torch.tensor(obstacles_map, dtype=torch.float32).unsqueeze(0)
    input = torch.cat((rewards, obstacles_map), dim=0)
    return input

def reformat_output(V):
    return torch.tensor(V, dtype=torch.float32).unsqueeze(0)


def generate_data(obstacle_map=None):
    """Generate map and target for value iteration 
    """
    rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map)
    neighbors = precompute_next_states(n,obstacles_map)    
    target = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
    target = reformat_output(target)
    input = reformat_input(rewards, obstacles_map)

    return input, target


def generate_policy_data():
    """Generate map and target for policy iteration
    """
    rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)
    neighbors = precompute_next_states(n,obstacles_map)    
    target = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
    policy = extract_policy(target, obstacles_map,neighbors)

    target = reformat_output(policy)
    input = reformat_input(rewards, obstacles_map)

    return input, target

class ValueIterationDataset(Dataset):
    """Value Iteration Dataset
    """
    def __init__(self, num_samples,obstacle_map=None):
        self.num_samples = num_samples
        self.obstacle_map = obstacle_map

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input, target = generate_data(self.obstacle_map)
        return input, target
    
class PolicyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input, target = generate_data()
        return input, target



class ValueIterationModel(torch.nn.Module):
    """Encoder-Decoder model for Value Iteration
    """
    def __init__(self):
        super(ValueIterationModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1) 
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x
    
class ValueIterationModelWithPooling(torch.nn.Module):
    """Encoder-Decoder model for Value Iteration
    """
    def __init__(self):
        super(ValueIterationModelWithPooling, self).__init__()
        # Encoder (Downsampling)
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downsample to (64, 5, 5)

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        # Use pooling only once to avoid excessive downsampling
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Avoid this second downsample
        
        # Decoder (Upsampling)
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = torch.nn.BatchNorm2d(1)

        self.final_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        # No pooling here to avoid reducing the spatial dimensions too much
        
        # Decoder
        x = F.relu(self.bn3(self.deconv1(x)))
        x = F.relu(self.bn4(self.deconv2(x)))

        # Final convolution to adjust to (1, 10, 10)
        x = self.final_conv(x)

        # Sigmoid activation to keep output between 0 and 1
        x = torch.sigmoid(x)
        return x



class DeeperValueIterationModel(torch.nn.Module):
    def __init__(self):
        super(DeeperValueIterationModel, self).__init__()
        
        # Encoder (Downsampling) Layers
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        
        # Decoder (Upsampling) Layers
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        
        # Sigmoid activation to keep output between 0 and 1
        x = torch.sigmoid(x)
        return x

class PolicyModel(torch.nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256,100)
        self.fc4 = torch.nn.Linear(100,4)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.softmax(x,dim=1) 
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x): 
        pass

def train_model(dataloader, model, model_path): 
    """Train the Value Iteration model"""
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    num_epochs = 10
    model.train()
    epoch_loss = []

    for epoch in range(num_epochs):
        acc_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, data in enumerate(dataloader):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                print("model out", outputs.shape)
                loss = criterion(outputs, targets)
                acc_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
        epoch_loss.append(acc_loss / len(dataloader))

    torch.save(model.state_dict(), model_path)

    plt.plot([i for i in range(num_epochs)], epoch_loss)
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

def eval_model(dataloader,model,model_path="deeper_value_iteration_model.pth"):
    """Evaluate the Value Iteration model
    """
    # model = ValueIterationModel()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()

    mse_loss = torch.nn.MSELoss()   
    total_loss = 0
    num_batches = 0 

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = mse_loss(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    average_mse = total_loss / num_batches
    return average_mse


if __name__ == "__main__":
    from eval import *
    num_samples = 50000

    with open('obstacle.pkl', 'rb') as f:
        f.seek(0) 
        obstacles_map = pickle.load(f)

    dataset = ValueIterationDataset(num_samples,obstacles_map)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model_path = "model_weights/value_function_fixed_map_3.pth"
    # train_model(dataloader,ValueIterationModelWithPooling(),model_path)
    train_model(dataloader,ValueIterationModel(),model_path)



  









    

