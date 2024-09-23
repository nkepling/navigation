import pickle
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from eval import get_vi_path
from nn_training import * 
import argparse
import tqdm
import fo_solver


# Define the input map
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



def get_full_trajectory(n, rewards, obstacles_map, neighbors, start):
    agent_position = deepcopy(start)
    steps = 0
    max_steps = 100
    path = [agent_position]
    reward_map_list = []

    # Ensure obstacles_map has the shape (1, n, n) for concatenation


    while np.any(rewards) and steps < max_steps:
        # Zero out the reward at the current agent's position
        rewards[agent_position[0], agent_position[1]] = 0
        
        # Append a copy of the rewards map at this step to avoid mutating it in future iterations
        reward_map_list.append(rewards.copy().reshape(1, n, n))
        
        # Run value iteration to get the value function and policy
        V = value_iteration(n, rewards, obstacles_map, gamma, neighbors)
        policy = extract_policy(V, obstacles_map, neighbors)
        
        # Update agent's position based on the policy
        next_position = tuple(int(i) for i in policy[agent_position])
        agent_position = next_position
        
        # Append new position to the path
        path.append(agent_position)
        steps += 1

    # Concatenate the reward maps with the obstacles map along the channel axis
    obstacles_map = np.where(obstacles_map, 1, 0).reshape(1, n, n)
    reward_map_list = [np.concatenate((img, obstacles_map), axis=0) for img in reward_map_list]

    return np.array(path), np.array(reward_map_list)

def extract_action(traj):
    """Given a trajectory, extract the actions that were taken.  The actions are used to train the VIN model.
    traj is a list of coordinates from start to goal. 
    """

    actions = []
    action_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
    state_diff = np.diff(traj, axis=0)
    for i in range(len(state_diff)):
        action = action_map[tuple(state_diff[i])]
        actions.append(action)

    assert len(actions) == len(traj)-1

    return np.array(actions)


def sample_trajectories(num_trajectories,reward,obstacle_map):
    """Grab trajectories from the dynamic programming solution to the value iteration problem.  The trajectories are used to train the VIN model.
    states_xy is a list off coordinates from start to goal. 
    """
    states_xy = []
    neighbors = precompute_next_states(n,obstacle_map)
    for i in range(num_trajectories):
        start,goal = pick_start_and_goal(reward,obstacle_map)
        path = get_vi_path(n, reward, obstacle_map, neighbors, start, goal)
        path = np.array(path)
        states_xy.append(path)
    
    # states_xy = np.array(states_xy)
    return states_xy






"""
Generate dataset for training the VIN model.  The inputs are images whre one channel encoodes the agent position and the other channel encodes the reward map.
The reward image encodes both the obstacles and the rewards map.
"""

def vin_data(n_rewards):
    with open("obstacle.pkl", "rb") as f:
        obstacle_map = pickle.load(f)

    X = []
    S1 = []
    S2 = []
    Labels = []

    neighbors = precompute_next_states(n,obstacle_map)

    with tqdm.tqdm(total=n_rewards) as pbar:
        for _ in range(n_rewards):
            reward,obstacle_map = init_map(n,config,num_blocks,num_obstacles,obstacle_type,obstacle_map=obstacle_map)
            # states_xy = sample_trajectories(num_trajectories,reward,obstacle_map) # shoould be numpy array of shape (num_trajectories, num_states, 2)
            states_xy,reward_list =get_full_trajectory(n,reward,obstacle_map,neighbors,start=(0,0))
        
          
            actions = extract_action(states_xy) # for each trajectory, extract the actions that were taken
            states_xy = states_xy[:-1] 
            assert reward_list.shape == (len(states_xy),2,n,n), f"reward_list shape {reward_list.shape}"
            ns = len(states_xy)
            # map_data = np.where(obstacle_map,1,0) # obstacle map 1 if obstacle, 0 if free
            # map_data = np.resize(map_data,(1,1,n,n))
            # value_prior = np.resize(reward,(1,1,n,n))
            # iv_mixed = np.concatenate((map_data,value_prior),axis=1)

            # X_cur  = np.tile(iv_mixed,(ns,1,1,1))
            S1_cur = np.expand_dims(states_xy[0:ns,0],axis=1) # x coordinates

            S2_cur = np.expand_dims(states_xy[0:ns,1],axis=1) # y coordinates

            Labels_cur = np.expand_dims(actions,axis=1) # actions taken

            X.append(reward_list)
            S1.append(S1_cur)
            S2.append(S2_cur)
            Labels.append(Labels_cur)
            pbar.update(1)
        
    X = np.concatenate(X,axis=0)
    S1 = np.concatenate(S1,axis=0)
    S2 = np.concatenate(S2,axis=0)
    Labels = np.concatenate(Labels,axis=0)

    print("X shape ",X.shape)
    print("S1 shape ",S1.shape)
    print("S2 shape ",S2.shape)
    print("Labels shape ",Labels.shape)



    return X,S1,S2,Labels

def main(n_train,n_test,save_path):
    os.makedirs("vin_data",exist_ok=True)
    print("Generating training data for VIN model")
    X,S1,S2,Labels = vin_data(n_train)
    print("Generating_test_data")
    X_test,S1_test,S2_test,Labels_test = vin_data(n_test)
    np.savez_compressed(save_path,X,S1,S2,Labels,X_test,S1_test,S2_test,Labels_test)

    print(f"Saved data to {save_path}")
    return X,S1,S2,Labels

if __name__ == "__main__":

    num_trajectories = 3 # number of trajectories to sample from each reward map
    #n_rewards = 3 # numerb of reward maps to generate
    n_train = 5000
    n_test = 1000
    save_path = "training_data/full_traj_vin_data.npz"
    X,S1,S2,Labels = main(n_train,n_test,save_path)
    # X,S1,S2,Labels = vin_data(num_trajectories,n_rewards)

    
    print("X ",X.shape)
    print("S1 ",S1.shape)
    print("S2 ",S2.shape)
    print("Labels ",Labels.shape)







