import pickle
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from eval import get_vi_path
from nn_training import * 
import argparse
import tqdm


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

def vin_data(num_trajectories,n_rewards):
    with open("obstacle.pkl", "rb") as f:
        obstacle_map = pickle.load(f)

    X = []
    S1 = []
    S2 = []
    Labels = []
    with tqdm.tqdm(total=n_rewards) as pbar:
        for _ in range(n_rewards):
            reward,obstacle_map = init_map(n,config,num_blocks,num_obstacles,obstacle_type,obstacle_map=obstacle_map)
            states_xy = sample_trajectories(num_trajectories,reward,obstacle_map) # shoould be numpy array of shape (num_trajectories, num_states, 2)
            for i in range(num_trajectories): # for each trajectory 
                actions = extract_action(states_xy[i]) # for each trajectory, extract the actions that were taken

                ns  = states_xy[i].shape[0] - 1 # number of states in the trajectory
                map_data = np.where(obstacle_map,1,0) # obstacle map 1 if obstacle, 0 if free
                map_data = np.resize(map_data,(1,1,n,n))
                value_prior = np.resize(reward,(1,1,n,n))
                iv_mixed = np.concatenate((map_data,value_prior),axis=1)

                X_cur  = np.tile(iv_mixed,(ns,1,1,1))
                S1_cur = np.expand_dims(states_xy[i][0:ns,0],axis=1)
                # S1_cur = states_xy[i][0:ns,0]
                S2_cur = np.expand_dims(states_xy[i][0:ns,1],axis=1)
                # S2_cur = states_xy[i][0:ns,1]
                Labels_cur = np.expand_dims(actions,axis=1)

                X.append(X_cur)
                S1.append(S1_cur)
                S2.append(S2_cur)
                Labels.append(Labels_cur)
            pbar.update(1)
        
    X = np.concatenate(X,axis=0)
    S1 = np.concatenate(S1,axis=0)
    S2 = np.concatenate(S2,axis=0)
    Labels = np.concatenate(Labels,axis=0)

    return X,S1,S2,Labels

def main(num_trajectories,n_rewards):
    os.makedirs("vin_data",exist_ok=True)
    save_path = f"vin_data/{n}x{n}_grid"
    print("Generating training data for VIN model")
    X,S1,S2,Labels = vin_data(num_trajectories,n_rewards)
    print("Generating_test_data")
    X_test,S1_test,S2_test,Labels_test = vin_data(num_trajectories,1000)
    np.savez_compressed(save_path,X,S1,S2,Labels,X_test,S1_test,S2_test,Labels_test)

    print(f"Saved data to {save_path}")
    return X,S1,S2,Labels

if __name__ == "__main__":

    num_trajectories = 3 # number of trajectories to sample from each reward map
    n_rewards = 5000 # numerb of reward maps to generate
    X,S1,S2,Labels = main(num_trajectories,n_rewards)
    # X,S1,S2,Labels = vin_data(num_trajectories,n_rewards)

    
    print("X ",X.shape)
    print("S1 ",S1.shape)
    print("S2 ",S2.shape)
    print("Labels ",Labels.shape)







