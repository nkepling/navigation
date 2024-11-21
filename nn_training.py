import torch.utils
from utils import *
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
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
from dl_models import UNet

# ensure_initialized()
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

def save_data_set(num_samples,obstacle_map=None,data_dir="value_iteration_data"):
    """Create the dataset for training value iteration model 
    """


    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Generate and save data samples
    id = 0
    neighbors = precompute_next_states(n,obstacle_map)
    with tqdm(total=num_samples, desc="Generating Data", unit="sample") as pbar:
        for i in range(num_samples):
            rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map)
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
        data = torch.load(os.path.join(self.data_dir, f"sample_{id}.pt"),weights_only=True)

        reward = data["reward"]
        value_map = data["value_map"]
        obstacles = data["obstacles"]



        # X = torch.cat((reward, obstacles), dim=0)

        # return X, value_map

        return reward, obstacles, value_map


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

class AutoEncoderDataset(Dataset):
    """AutoEncoder Dataset
    """
    def __init__(self, data_dir, indices):
        """
        Args:
            data_dir (str): Path to the directory containing the saved .pt files.
            indices (list): List of indices for the dataset (used to load specific samples).
        """
        self.data_dir = data_dir
        self.indices = indices

    def __len__(self):
        """Return the length of the dataset (number of samples)."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Load the input, coordinates, and action from the .pt file for the given index.
        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            input (Tensor): The input tensor for the autoencoder.
            coords (Tensor): The corresponding coordinates for the input.
            action (Tensor): The action tensor.
        """
        id = self.indices[idx]
        file_path = os.path.join(self.data_dir, f"map_{id}.pt")
        
        # Load the input, coordinates, and action from the .pt file
        input, coords, action,V = torch.load(file_path,weights_only=True)
        input = input.squeeze(1)

        V = V.view(1,10,10)

        
        
        assert input.shape == V.shape,(f"Got input shape {input.shape} and v shape {V.shape}")
        return input.float(), coords.float(), action, V.float()
    




class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, warmup_epochs=10, dir_path="model_weights/value_cnn"):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs  # Wait this many epochs before applying early stopping
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = dir_path

    def __call__(self, epoch, val_loss, model):
        if epoch < self.warmup_epochs:
            # Don't apply early stopping during the warm-up period
            return

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model,epoch)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model,epoch)
            self.counter = 0

    def save_checkpoint(self, model,epoch):
        '''Save the model when validation loss improves.'''
        torch.save(model.state_dict(), self.path+f"checkpoint_epoch_{epoch}.pth")
        print(f"Model saved at epoch with validation loss: {self.best_loss:.4f}")



def validate(model, validation_loader, criterion1, device):
    model.eval()  # Set the model to evaluation mode

    v_loss = []
    
    with torch.no_grad():  # Ensure no gradients are computed
        for data in validation_loader:
            X,_,V  = data
            V = V.unsqueeze(1)
            X, V = X.to(device), V.to(device)

            # Forward pass
            outputs = model(X)
            
            # Calculate loss

            assert outputs.shape == V.shape, f"Got {outputs.shape} , {V.shape}"
            loss = criterion1(outputs, V)
            v_loss.append(loss.item())
            
    
    # Calculate average loss and accuracy
    avg_loss = np.mean(v_loss)

    return avg_loss

def train_model(train_dataloader, val_dataloader, model, model_path,num_epochs,criterion,delta): 
    """Train the Value Iteration model"""
    model = model.to(device)
    # criterion1 = torch.nn.MSELoss()

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    model.train()
    train_epoch_loss = []
    val_loss = []
    
    early_stopper = EarlyStopping(patience=10,delta=delta,warmup_epochs=15,dir_path=model_path)

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar0:
        for epoch in range(num_epochs):
            acc_loss = 0
 
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, data in enumerate(train_dataloader):

                    
                    # inputs, coords,action,V = data
                    # inputs, V = inputs.float().to(device), V.float().to(device)

                    X, obstacles, value_map = data

                    value_map = value_map.unsqueeze(1)

                    X = X.to(device)
                    value_map = value_map.to(device)





                    optimizer.zero_grad()
                    outputs = model(X)

                    assert outputs.shape == value_map.shape, f"Got output shape of {outputs.shape} and V shape of {value_map.shape}"
                    loss = criterion(outputs, value_map)
                    acc_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)


            train_epoch_loss.append(acc_loss / len(train_dataloader))
            v_loss = validate(model, val_dataloader, criterion, device)
            val_loss.append(v_loss)

            pbar0.set_postfix(v_loss=v_loss)
            pbar0.update(1)

            if epoch%50==0:
                path = model_path + f"checkpoint_epoch_{epoch}.pt"
                torch.save(model.state_dict(), path)

            early_stopper(epoch,v_loss,model)
            if early_stopper.early_stop:
                print("Early stop at epoch: ",epoch)
                break

    torch.save(model.state_dict(), model_path + "final_model.pt")

    plt.plot([i for i in range(len(train_epoch_loss))], train_epoch_loss)
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("AE Training Loss")
    plt.savefig("images/value_cnn/train_loss.png")

    plt.plot([i for i in range(len(val_loss))], val_loss)    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.savefig("images/value_cnn/val_loss.png")

    print("DONE!!!")



def eval_model(dataloader,model,model_path):
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
    model.train()
    return average_mse




def autoencoder_val(model,val_data_loader,criterion,device):
    model.eval()
    val_loss = 0
    for i,data in enumerate(val_data_loader):
        inputs,coords,action = data
        inputs = inputs.to(device)
        inputs.requires_grad_(True)
        recon,latent = model(inputs)

        loss = criterion(inputs,recon,latent)
        val_loss+= loss.item()

    return val_loss/len(val_data_loader)
        

        
    

def train_auto_encoder(train_dataloader,val_dataloader, model,model_path,num_epochs,criterion,optimizer):
    model = model.to(device)

    model.train()
    train_loss = []
    val_loss = []


    early_stopper = EarlyStopping(patience=10,delta=0.001,warmup_epochs=15,path=model_path)

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar0:
            for epoch in range(num_epochs):
                acc_loss = 0 
                with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                    for i, data in enumerate(train_dataloader):
                        inputs, coords,action,V = data

                        inputs  =  inputs.to(device)
                        inputs.requires_grad_(True)

                        optimizer.zero_grad()
                        recon,latent = model(inputs)

                        loss = criterion(inputs, recon,latent) 
                        acc_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(loss=loss.item())
                        pbar.update(1)


                train_loss.append(acc_loss / len(train_dataloader))
                v_loss = autoencoder_val(model, val_dataloader, criterion, device)

                pbar0.set_postfix(v_loss=v_loss)
                pbar0.update(1)
                if epoch%10==0:
                    path = f"model_weights/CAE_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), path)
                val_loss.append(v_loss)

                early_stopper(epoch,v_loss,model)
                if early_stopper.early_stop:
                    print("Early stop at epoch: ",epoch)
                    break

                

    torch.save(model.state_dict(),model_path)
    plt.figure()
    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("AE Training Loss")
    plt.savefig("images/CAE_train_loss.png")

    plt.figure()
    plt.plot([i for i in range(len(val_loss))], val_loss)    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.savefig("images/CAE_val_loss.png")
            
def compare_images(input_image, recon_image):
    """
    Display the input and reconstructed images side by side with a shared color scale and color bar.
    
    Args:
        input_image: The original input image as a numpy array.
        recon_image: The reconstructed image as a numpy array.
    """
    # Convert to numpy arrays and remove any extra dimensions
    input_image = input_image.cpu().detach().numpy().squeeze()
    recon_image = recon_image.cpu().detach().numpy().squeeze()

    # Mask -1 values (if present)
    input_image_masked = np.where(input_image == -1, np.nan, input_image)
    recon_image_masked = np.where(input_image == -1, np.nan, recon_image)

    # Determine the shared color scale limits (vmin and vmax)
    vmin = min(np.nanmin(input_image_masked), np.nanmin(recon_image_masked))
    vmax = max(np.nanmax(input_image_masked), np.nanmax(recon_image_masked))

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Input image with color bar
    img1 = axes[0].imshow(input_image_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Reconstructed image with color bar
    img2 = axes[1].imshow(recon_image_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    fig.colorbar(img1, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)


    # Save the figure
    plt.savefig("images/autoencoder_comparison.png")

def validate_pnet(val_dataloader, pnet_model, CAE_model, criterion, device):
    """
    Validate the PNet model using the validation data.
    
    Args:
        val_dataloader: DataLoader for the validation data.
        pnet_model: The PNet model being validated.
        CAE_model: Pre-trained CAE model for extracting latent representations.
        criterion: Loss function (cross-entropy).
        device: Device to use ('cuda' or 'cpu').

    Returns:
        float: The average validation loss.
    """
    pnet_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, coords, actions in val_dataloader:
            inputs = inputs.to(device)
            coords = coords.to(device)
            actions = actions.to(device)

            # Extract latent representation from frozen CAE model
            _, latent = CAE_model(inputs)

            # Forward pass through PNet
            outputs = pnet_model(coords, latent)

            # Compute validation loss
            loss = criterion(outputs, actions)
            total_val_loss += loss.item()

    return total_val_loss / len(val_dataloader)

def test_acc_pnet(val_dataloader, pnet_model, CAE_model, device):
    pnet_model.eval()
    CAE_model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, coords, actions in val_dataloader:
            inputs = inputs.to(device)
            coords = coords.to(device)
            actions = actions.to(device)

            # Get latent representations from the CAE model
            _, latent = CAE_model(inputs)

            # Forward pass through PNet
            outputs = pnet_model(coords, latent)

            # Get predictions by taking the argmax of the logits
            pred = torch.argmax(outputs, dim=1)

            # Calculate the number of correct predictions
            correct = (pred == actions).sum().item()  # Sum the number of correct predictions

            # Update total counts
            total_correct += correct
            total_samples += actions.size(0)  # Update the number of total samples

    # Calculate overall accuracy
    accuracy = total_correct / total_samples
    return accuracy

            



def train_pnet(train_dataloader, val_dataloader, pnet_model, CAE_model, CAE_model_path, pnet_model_path, num_epochs, criterion, optimizer):
    """
    Train the PNet model using coordinates and latent representations from a pre-trained CAE model.

    Args:
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.
        pnet_model: The PNet model to be trained.
        CAE_model: Pre-trained CAE model to extract latent representations.
        pnet_model_path: Path to save the trained PNet model.
        num_epochs: Number of epochs for training.
        criterion: Loss function (cross-entropy for action prediction).
        optimizer: Optimizer for PNet.
    """
    pnet_model = pnet_model.to(device)
    CAE_model.load_state_dict(torch.load(CAE_model_path,weights_only=True))
    CAE_model = CAE_model.to(device)
    CAE_model.eval()  # Freeze CAE model during PNet training

    train_loss = []
    val_loss = []

    # Initialize early stopping mechanism
    early_stopper = EarlyStopping(patience=10, delta=0.001, warmup_epochs=15, path=pnet_model_path)

    with tqdm(total=num_epochs, desc="Training PNet", unit="epoch") as pbar0:
        for epoch in range(num_epochs):
            acc_loss = 0  # Accumulated training loss
            pnet_model.train()

            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, data in enumerate(train_dataloader):
                    inputs, coords, actions = data

                    inputs = inputs.to(device)
                    coords = coords.to(device)
                    actions = actions.to(device)

                    # Freeze CAE and extract latent representations
                    with torch.no_grad():
                        _, latent = CAE_model(inputs)

                    optimizer.zero_grad()

                    # Forward pass through PNet
                    outputs = pnet_model(coords, latent)

                    # Compute cross-entropy loss
                    loss = criterion(outputs, actions)
                    acc_loss += loss.item()

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            # Compute average training loss
            train_loss.append(acc_loss / len(train_dataloader))

            # Validation step
            v_loss = validate_pnet(val_dataloader, pnet_model, CAE_model, criterion, device)
            val_loss.append(v_loss)

            pbar0.set_postfix(v_loss=v_loss)
            pbar0.update(1)

            # Save PNet model periodically
            if epoch % 10 == 0:
                path = f"model_weights/PNetResNet_epoch_{epoch}.pth"
                torch.save(pnet_model.state_dict(), path)

            # Early stopping check
            early_stopper(epoch, v_loss, pnet_model)
            if early_stopper.early_stop:
                print(f"Early stop at epoch {epoch}")
                break

    # Save the final model
    torch.save(pnet_model.state_dict(), pnet_model_path)

    # Plot training and validation loss
    plt.figure()
    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PNet Training Loss")
    plt.savefig("images/PNetResNet2_train_loss.png")

    plt.figure()
    plt.plot([i for i in range(len(val_loss))], val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.savefig("images/PNetRestNet2_val_loss.png")



def dagger(expert_policy,expert_data_set,model,num_iters):
    """Data Aggregation for imiation learning

    NOTE: This looks like it will be computationally expensive. 
    """

    for _ in range(num_iters):
        pass




def main(config):

    train_directory = config.train_data_dir
    train_file_count = sum(1 for f in os.listdir(train_directory) if os.path.isfile(os.path.join(train_directory, f)))
    print("train_file_count", train_file_count)
    # train_file_count = 2

    train_dataloader = DataLoader(SavedData(config.train_data_dir,indeces=[i for i in range(train_file_count)]),batch_size=config.batch_size,shuffle=True)
    
    val_directory = config.val_data_dir
    val_file_count = sum(1 for f in os.listdir(val_directory) if os.path.isfile(os.path.join(val_directory, f)))
    print("val_file_count", val_file_count)
    # val_file_count = 2

    val_dataloader = DataLoader(SavedData(config.val_data_dir,indeces=[i for i in range(val_file_count)]),batch_size=config.batch_size,shuffle=True)

    model = UNet()
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    num_epochs = config.num_epochs

    delta = config.delta
    model_path = config.model_path_dir

    train_model(train_dataloader, val_dataloader, model, model_path,num_epochs,criterion,delta)

      

if __name__ == "__main__":
    import argparse
    
    
    
    
    parser = argparse.ArgumentParser()

    # DATA
    
    parser.add_argument("--train_data_dir", type=str, default="training_data/value_cnn_train", help="Directory containing the training data.")
    parser.add_argument("--val_data_dir", type=str, default="training_data/value_cnn_test", help="Directory containing the validation data.")
    

    # Training params

    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--delta", type=float, default=0.001, help="Delta for early stopping.")
    parser.add_argument("--model_path_dir", type=str, default="model_weights/value_cnn/", help="Path to save the trained model.")

    
    config = parser.parse_args()

    main(config)


 













    









    
