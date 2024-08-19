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
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
    obstacles_map = torch.tensor(obstacles_map, dtype=torch.float32).unsqueeze(0)
    input = torch.cat((rewards, obstacles_map), dim=0)
    return input

def reformat_output(V):
    return torch.tensor(V, dtype=torch.float32).unsqueeze(0)


def generate_data():
    """Generate map and target for value iteration 
    """
    rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)
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
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input, target = generate_data()
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



def train_model(dataloader,model): 
    """Train the Value Iteration model
    """
    # model = ValueIterationModel().to(device)
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-3)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "deeper_value_iteration_model.pth")

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
    num_samples = 10000
    dataset = ValueIterationDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for idx, data in enumerate(dataloader):
        inputs, targets = data
        print(inputs.shape, targets.shape)
        break

    # train_model(dataloader,DeeperValueIterationModel())
    # average_mse = eval_model(dataloader,DeeperValueIterationModel(),model_path="deeper_value_iteration_model.pth")
    # print(f"Average MSE Loss: {average_mse}")


    

    """ Navigation """
    # model = ValueIterationModel()
    # model.load_state_dict(torch.load("value_iteration_model.pth",weights_only=True))
    # model.eval()

    model = DeeperValueIterationModel()
    model.load_state_dict(torch.load("deeper_value_iteration_model.pth",weights_only=True))
    model.eval()
    rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)
    neighbors = precompute_next_states(n, obstacles_map)

    start, goal = pick_start_and_goal(rewards, obstacles_map)
    visualize_rewards(rewards, obstacles_map, start, goal)

    if start == goal:
        print("the agent is already in the target position")

    agent_position = deepcopy(start)
    while agent_position!=goal:
        # mark current position as 0 reward
        rewards[agent_position[0], agent_position[1]] = 0

        input = reformat_input(rewards, obstacles_map)
        V = model(input)

        V = V.squeeze().detach().numpy()
        policy = extract_policy(V, obstacles_map,neighbors)
        
        next_position = tuple(int(i) for i in policy[agent_position])
        print("Agent next state is {}".format(next_position))
        i, j = agent_position[0], agent_position[1]
        # visualize_rewards(rewards, obstacles_map, start, goal, agent_position, next_position)
        visualize_policy_and_rewards(rewards, obstacles_map, policy)
        agent_position = next_position










    

