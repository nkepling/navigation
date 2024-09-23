import torch
import numpy as np
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.data import Data
import torch.nn.functional as F
from gnn_utils import *
from nn_training import train_model
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = geom_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = geom_nn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
def validate_gnn(model, validation_loader, criterion1, device):
    model.eval()  # Set the model to evaluation mode

    v_loss = []
    
    with torch.no_grad():  # Ensure no gradients are computed
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs.x, inputs.edge_index)
            
            # Calculate loss
            loss1 = criterion1(outputs, labels)

            loss = loss1 
            v_loss.append(loss.item())
            
    
    # Calculate average loss and accuracy
    avg_loss = np.mean(v_loss)

    return avg_loss

def train_gnn_model(train_dataloader, val_dataloader, model, model_path,num_epochs=20): 
    """Train the Value Iteration model"""
    model = model.to(device)
    criterion1 = torch.nn.MSELoss()

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.1)
    model.train()
    train_epoch_loss = []
    val_epoch_loss = []
    with tqdm(total=num_epochs, desc="Training", unit="epoch") as pbar0:
        for epoch in range(num_epochs):
            acc_loss = 0
 
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, data in enumerate(train_dataloader):
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs.x, inputs.edge_index)
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
            val_loss = validate_gnn(model, val_dataloader, criterion1, device)
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

if __name__ == "__main__":

    model = GCN(2, 64, 1)


    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # criterion = torch.nn.MSELoss()

    train_ind,val = torch.utils.data.random_split(range(100000),[80000,20000])
    train_data = SavedGraphData(data_dir="graph_data",indeces=train_ind)
    # val_data = SavedGraphData(data_dir="graph_data",indeces=val)

    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)

    for data in train_loader:
        inputs, targets = data
        print(inputs.num_nodes)
        print(inputs.edge_index.shape)
        out = model(inputs.x,inputs.edge_index)
        print(out.shape)
        print(targets.shape)
        break
    # val_loader = DataLoader(val_data,batch_size=32,shuffle=True)

    # train_gnn_model(train_loader, val_loader, model, "gcn_model.pth",num_epochs=20)

