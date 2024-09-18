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
        input, target = torch.load(os.path.join(self.data_dir, f"map_{id}.pt"),weights_only=True)
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
    def __init__(self, patience=10, delta=0.001, warmup_epochs=10, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.warmup_epochs = warmup_epochs  # Wait this many epochs before applying early stopping
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, epoch, val_loss, model):
        if epoch < self.warmup_epochs:
            # Don't apply early stopping during the warm-up period
            return

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Save the model when validation loss improves.'''
        torch.save(model.state_dict(),"model_weights/" + self.path+".pth")
        print(f"Model saved at epoch with validation loss: {self.best_loss:.4f}")



def validate(model, validation_loader, criterion1, device):
    model.eval()  # Set the model to evaluation mode

    v_loss = []
    
    with torch.no_grad():  # Ensure no gradients are computed
        for data in validation_loader:
            inputs, coords, action,V = data
            inputs, labels = inputs.to(device), V.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss1 = criterion1(outputs, labels)

            loss = loss1 
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
    
    early_stopper = EarlyStopping(patience=10,delta=delta,warmup_epochs=15,path=model_path)

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar0:
        for epoch in range(num_epochs):
            acc_loss = 0
 
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, data in enumerate(train_dataloader):
                    inputs, coords,action,V = data
                    inputs, V = inputs.float().to(device), V.float().to(device)



                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, V)
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

            if epoch%10==0:
                path = "model_weights/" + f"{model_path}_{epoch}.pth"
                torch.save(model.state_dict(), path)

            early_stopper(epoch,v_loss,model)
            if early_stopper.early_stop:
                print("Early stop at epoch: ",epoch)
                break

    torch.save(model.state_dict(), model_path)

    plt.plot([i for i in range(len(train_epoch_loss))], train_epoch_loss)
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("AE Training Loss")
    plt.savefig("images/"+model_path+"train_loss.png")

    plt.plot([i for i in range(len(val_loss))], val_loss)    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.savefig("images/"+model_path+"val_loss.png")

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



      

if __name__ == "__main__":
    pass
    # from dl_models import UNetSmall,CAE_Loss,ContractiveAutoEncoder
    # from utils import *
    # import pickle
    # from dl_models import *




 



    # dataset_size =len([os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('pt'))])

    # train_size = int(0.7 * dataset_size)  # 70% for training
    # val_size = int(0.15 * dataset_size)   # 15% for validation
    # test_size = dataset_size - train_size - val_size  # 15% for testing
   
    # train_ind,test_ind,val_ind = torch.utils.data.random_split(range(dataset_size),[train_size,test_size,val_size])

    # train_data = AutoEncoderDataset(data_dir,train_ind,num_workers=4,pin_memory=True)
    # val_data = AutoEncoderDataset(data_dir,val_ind,num_workers=4,pin_memory=True)
    # test_data = AutoEncoderDataset(data_dir,test_ind,num_workers=4,pin_memory=True)

    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_data,batch_size=1,shuffle=True)
    
    # # # model = UNetSmall()
    # # model = UNetSmall()
    # # model_path = "model_weights/unet_small_7.pth"

    # # train_model(train_loader,val_loader, model, model_path,num_epochs=80)

    # # # print(eval_model(test_loader,model=model, model_path=model_path))

    # cae_model = ContractiveAutoEncoder()
    # model_path = "model_weights/CAE_1.pth"
    # # # criterion = CAE_Loss(beta=1e-4)

    # #  # Adjust lr based on your needs

    # # # pnet_model = PNet(2,latent_dim=128,hidden_dim=128)
    # # # pnet_model_path = "model_weights/pnet_1.pth"
    # pnet_model = PNetResNet(coord_dim=2,latent_dim=128,hidden_dim=128,num_blocks=8) # the other has 5 blocks
    # pnet_model_path = "model_weights/pnet_resnet_2.pth"
    # # optimizer = torch.optim.Adam(pnet_model.parameters(), lr=1e-4)

    # # criterion = torch.nn.CrossEntropyLoss()
    
    # #train_pnet(train_loader,val_loader,pnet_model,cae_model,model_path,pnet_model_path,num_epochs=100,criterion=criterion,optimizer=optimizer)
    

    # # train_auto_encoder(train_dataloader=train_loader,
    # #                    val_dataloader=val_loader,
    # #                    model=model,
    # #                    model_path=model_path,
    # #                    num_epochs=100,
    # #                    criterion=criterion,
    # #                    optimizer=optimizer)
    # # model.load_state_dict(torch.load(model_path,weights_only=True))
    # # model.to(device)
    # #print(autoencoder_val(model,test_loader,criterion=criterion,device=device))

    # pnet_model.load_state_dict(torch.load(pnet_model_path,weights_only=True))

    # cae_model.load_state_dict(torch.load(model_path,weights_only=True))
    # cae_model.to(device)
    # pnet_model.to(device)

    # # # print(validate_pnet(test_loader,pnet_model=pnet_model,CAE_model=cae_model))

    # print(test_acc_pnet(val_dataloader=test_loader,pnet_model=pnet_model,CAE_model=cae_model,device=device))

    # # # Get one item from the test loader
    # # test_iter = iter(test_loader)
    # # input_image = next(test_iter)  # Assuming you only need the image, not the label

    # # im = input_image[0]

    # # # Move the input image to the correct device
    # # im = im.to(device)

    # # # Forward pass through the model

    # # recon, latent = model(im)


    # # compare_images(im,recon)
        
    










    









    
