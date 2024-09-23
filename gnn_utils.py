import torch
import numpy as np
import random
from utils import *
import pickle
from torch.utils.data import Dataset
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.data import Data
import os
from fo_solver import value_iteration
from tqdm import tqdm




def get_edge_indices(grid_size):
    sources = []
    targets = []
    for i in range(grid_size):
        for j in range(grid_size):
            node_idx = i * grid_size + j
            if i > 0:  # Connect to the node above
                sources.append(node_idx)
                targets.append(node_idx - grid_size)
            if i < grid_size - 1:  # Connect to the node below
                sources.append(node_idx)
                targets.append(node_idx + grid_size)
            if j > 0:  # Connect to the node to the left
                sources.append(node_idx)
                targets.append(node_idx - 1)
            if j < grid_size - 1:  # Connect to the node to the right
                sources.append(node_idx)
                targets.append(node_idx + 1)

    edge_index = torch.tensor([sources,targets],dtype=torch.long)
    return edge_index



def get_graph_data(n,obstacle_map,rewards,edges):
    """Get the graph data for the value iteration model GNN approx. 
    """
    
 
    obstacles_map = np.where(obstacle_map,1,0)

    x = torch.tensor([[rewards[i,j],obstacles_map[i,j]] for i in range(n) for j in range(n)],dtype=torch.float32) # node feature matirx (num_nodes, num_node_features)
    # edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edges)
    return data

    # x = torch.tensor(np.vstack([is_wall,rewards]).T,dtype=torch.float32)


def create_graph_data_set(num_samples,n,obstacles_map,data_dir="graph_data"):
    """Create a dataset of graph data for the value iteration model GNN approx. 
    """
    os.makedirs(data_dir, exist_ok=True)
    edges = get_edge_indices(10)
    neighbors = precompute_next_states(n,obstacles_map)
    with tqdm(total=num_samples, desc="Generating Data", unit="sample") as pbar:
        for i in range(num_samples):
            rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacles_map)
            data = get_graph_data(n,obstacles_map,rewards,edges)
            V = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
            v_graph = torch.tensor([V[i,j] for i in range(n) for j in range(n)],dtype=torch.float32)
            torch.save((data,v_graph),os.path.join(data_dir,f"graph_data_{i}.pt"))
            pbar.update(1)


def convert_values_to_graph(V,edges):
    x = torch.tensor([V[i,j] for i in range(n) for j in range(n)],dtype=torch.float32) # node feature matirx (num_nodes, num_node_features)
    edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data


class SavedGraphData(Dataset):
    """Value Iteration Dataset
    """
    def __init__(self,data_dir,indeces):

        self.data_dir = data_dir
        self.indeces = indeces

    def __len__(self):
        
        return len(self.indeces)

    def __getitem__(self, idx):
        id = self.indeces[idx]
        input, target = torch.load(os.path.join(self.data_dir, f"graph_data_{id}.pt"))
        return input, target


if __name__ == "__main__":
    from fo_solver import visualize_rewards

    with open("obstacle.pkl","rb") as f:
        obstacle_map = pickle.load(f)

    create_graph_data_set(100000,10,obstacle_map)
    # edge_ibdex = get_edge_indices(10)
    # print(edge_ibdex.shape)






