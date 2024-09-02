import torch.utils
from utils import *
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import numpy as np
import random
import torch.nn.functional as F
from fo_solver import value_iteration, extract_policy, visualize_policy_and_rewards, visualize_rewards, pick_start_and_goal
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# NOTE: You have to from utils import * in order to run this code. In the utils file I set the map + value iteration configs. 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class BellmanResidualLoss(nn.Module):
    def __init__(self):
        super(BellmanResidualLoss, self).__init__()

    def forward(self, X, Y):

        max_val1 = torch.max(X)
        max_val2 = torch.max(Y)
        
        loss = torch.abs(max_val1 - max_val2)
        
        return loss


def reformat_input(rewards, obstacles_map):
    """Reformat the input for the NN model
    """
    temp = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
    # obstacles_map  = np.where(obstacles_map,-1,0)
    obstacles_map = torch.tensor(obstacles_map, dtype=torch.float32).unsqueeze(0)
    input = torch.cat((temp, obstacles_map), dim=0)
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

def save_data_set(num_samples,obstacle_map=None):
    """Create the dataset for training value iteration model 
    """
    data_dir = 'value_iteration_data'

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Generate and save data samples
    id = 0
    with tqdm(total=num_samples, desc="Generating Data", unit="sample") as pbar:
        for i in range(num_samples):
            rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map)
            neighbors = precompute_next_states(n,obstacles_map)
            input = reformat_input(rewards, obstacles_map)
            V = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
            target = reformat_output(V)
            torch.save((input, target), os.path.join(data_dir, f"map_{id}.pt"))
            id += 1
            # start, goal = pick_start_and_goal(rewards, obstacles_map)
            # agent_position = deepcopy(start)
            V_prev = V

            # while agent_position!=goal:
            #     rewards[agent_position[0], agent_position[1]] = 0
            #     V = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
            #     policy = extract_policy(V, obstacles_map,neighbors)
            #     next_position = tuple(int(i) for i in policy[agent_position])
            #     agent_position = next_position


            #     torch.save((input, target), os.path.join(data_dir, f"map_{id}.pt"))
            #     id += 1

            pbar.update(1)

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

class SavedData(Dataset):
    """Value Iteration Dataset
    """
    def __init__(self,data_dir,indeces):

        self.data_dir = data_dir
        self.indeces = indeces

    def __len__(self):
        
        return len(self.indeces)

    def __getitem__(self, idx):
        id = self.indeces[idx]
        input, target = torch.load(os.path.join(self.data_dir, f"map_{id}.pt"))
        return input, target


class ValueIterationDataset(Dataset):
    """Value Iteration Dataset
    """
    def __init__(self, num_samples,obstacle_map=None,seeds=None):
        self.num_samples = num_samples
        self.obstacle_map = obstacle_map
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input, target = generate_data(self.obstacle_map)
        return input, target
    



def validate(model, validation_loader, criterion1, device):
    model.eval()  # Set the model to evaluation mode

    v_loss = []
    
    with torch.no_grad():  # Ensure no gradients are computed
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss1 = criterion1(outputs, labels)

            loss = loss1 
            v_loss.append(loss.item())
            
    
    # Calculate average loss and accuracy
    avg_loss = np.mean(v_loss)

    return avg_loss

def train_model(train_dataloader, val_dataloader, model, model_path): 
    """Train the Value Iteration model"""
    model = model.to(device)
    criterion1 = torch.nn.MSELoss()

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=150,gamma=0.1)
    num_epochs = 200
    model.train()
    train_epoch_loss = []
    val_epoch_loss = []
    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar0:
        for epoch in range(num_epochs):
            acc_loss = 0
 
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, data in enumerate(train_loader):
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    l1 = criterion1(outputs, targets)
                    loss = l1 
                    acc_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            pbar0.update(1)
            train_epoch_loss.append(acc_loss / len(train_dataloader))
            val_loss = validate(model, val_dataloader, criterion1, device)
            val_epoch_loss.append(val_loss)
            pbar0.set_postfix_str(f"Epoch Loss: {train_epoch_loss[-1]}, Validation Loss: {val_loss}")

    torch.save(model.state_dict(), model_path)

    plt.plot([i for i in range(num_epochs)], train_epoch_loss)
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    plt.plot([i for i in range(num_epochs)], val_epoch_loss)    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
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
    from dl_models import UNetSmall
    from utils import *
    import pickle

    # data = ValueIterationDataset(num_samples=num_samples)
    train_ind,val = torch.utils.data.random_split(range(100000),[80000,20000])

    train_data = SavedData(data_dir="value_iteration_data",indeces=train_ind)
    val_data = SavedData(data_dir="value_iteration_data",indeces=val)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    
    model = UNetSmall()
    model_path = "model_weights/unet_small_value_iteration_model_3.pth"

    train_model(train_loader,val_loader, model, model_path)








    









    

