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
min_obstacles = 2
max_obstacles = 10
n = 20
num_blocks = 5
square_size = 10    

# Discount factor
gamma = 0.9

# define experiment configuration
# random_map = True

def get_full_trajectory(n, rewards, obstacles_map, neighbors, start):
    agent_position = deepcopy(start)
    steps = 0
    max_steps = 1000
    path = [agent_position]
    reward_map_list = []
    value_map_list = []

    # Ensure obstacles_map has the shape (1, n, n) for concatenation


    while np.any(rewards) and steps < max_steps:
        # Zero out the reward at the current agent's position
        rewards[agent_position[0], agent_position[1]] = 0
        
        # Append a copy of the rewards map at this step to avoid mutating it in future iterations
        reward_map_list.append(rewards.copy().reshape(1, n, n))
        
        # Run value iteration to get the value function and policy
        
        V = value_iteration(n, rewards, obstacles_map, gamma, neighbors)
        value_map_list.append(V)
        policy = extract_policy(V, obstacles_map, neighbors,n)
        
        # Update agent's position based on the policy
        next_position = tuple(int(i) for i in policy[agent_position])
        agent_position = next_position
        
        # Append new position to the path
        path.append(agent_position)
        steps += 1

    # Concatenate the reward maps with the obstacles map along the channel axis
    obstacles_map = np.where(obstacles_map, 1, 0).reshape(1, n, n)
    reward_map_list = [np.concatenate((img, obstacles_map), axis=0) for img in reward_map_list]

    return np.array(path), np.array(reward_map_list),value_map_list

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
def vin_data(n_rewards, seeds, save_dir, num_reward_variants=7,):
    X = []
    ValueMaps = []  # To store value maps

    id = 0

    with tqdm.tqdm(total=len(seeds)) as pbar_obs:
        for seed in seeds:
            reward, obstacle_map = init_random_reachable_map(
                n, "block", 5, min_obstacles, max_obstacles,
                obstacle_type="block", square_size=25,
                obstacle_map=None, seed=seed,
                num_reward_blocks=(2, 20),
                reward_square_size=(2, 20),
                obstacle_cluster_prob=0.4,
                obstacle_square_sizes=(3, 20)
            )
            
            neighbors = precompute_next_states(n, obstacle_map)

            for variant in range(num_reward_variants):
                reward, _ = init_random_reachable_map(
                    n, "block", 5, min_obstacles, max_obstacles,
                    obstacle_type="block", square_size=25,
                    obstacle_map=obstacle_map, seed=seed,
                    num_reward_blocks=(2, 20),
                    reward_square_size=(2, 20),
                    obstacle_cluster_prob=0.4,
                    obstacle_square_sizes=(3, 20)
                )

                if np.sum(reward) == 0:
                    print("No reward, skipping.")
                    continue

                # Generate trajectories and value maps
                states_xy, reward_list,value_maps = get_full_trajectory(
                    n, reward.copy(), obstacle_map, neighbors, start=(0, 0)
                )

                if len(reward_list) == 0:
                    print(f"Skipping empty trajectory for seed {seed}, variant {variant}")
                    continue

                # # Compute value maps for each reward map in the trajectory
                # value_maps = []
                # for reward_snapshot in reward_list:
                #     value_map = value_iteration(n, reward_snapshot[0], obstacle_map, gamma, neighbors)
                #     value_maps.append(value_map)

      

                for ind,reward in enumerate(reward_list):
                    file_name = f"sample_{id}.pt"
                    file_path = os.path.join(save_dir, file_name)

                    torch.save({
                        "reward":torch.tensor(reward,dtype=torch.float32),
                        "value_map":torch.tensor(value_maps[ind],dtype=torch.float32),
                        "obstacles":torch.tensor(obstacle_map,dtype=torch.float32)
                    },file_path)

                    id+=1


            pbar_obs.update(1)



# def main(n_train, n_test, save_path, train_seeds, test_seeds, num_reward_variants=7):
#     os.makedirs("vin_data", exist_ok=True)
#     print("Generating training data for VIN model")

#     X, ValueMaps = vin_data_with_value_maps(
#         n_train, train_seeds, num_reward_variants
#     )

#     print("Generating test data")
#     X_test, ValueMaps_test = vin_data_with_value_maps(
#         n_test, test_seeds, num_reward_variants
#     )

#     # Save all data to a compressed file
#     np.savez_compressed(
#         save_path,
#         X=X,  ValueMaps=ValueMaps,
#         X_test=X_test, ValueMaps_test=ValueMaps_test
#     )

#     print(f"Saved data to {save_path}")
#     return X, ValueMaps



def main(n_train, n_test, save_dir, train_seeds,test_seeds,num_reward_variants=7):
    os.makedirs("vin_data", exist_ok=True)
    print("Generating training data for VIN model")
    vin_data(n_train, train_seeds, "training_data/value_cnn_train", num_reward_variants)  # 10 reward variants per map config
    print("Generating test data")
    vin_data(n_test, test_seeds, "training_data/value_cnn_test", num_reward_variants)
    # X_test, Values_test = vin_data(n_test,test_seeds, num_reward_variants=10)  # 10 reward variants per map config
    # np.savez_compressed(save_path, X,Values,)

    print(f"Saved data to {save_path}")
    # return X, S1, S2, Labels
if __name__ == "__main__":

    num_trajectories = 3 # number of trajectories to sample from each reward map
    #n_rewards = 3 # numerb of reward maps to generate
    n_train = 1000
    n_test = 500
    #n_rewards = 3

    train_seeds = [x for x in range(n_train)]
    test_seeds = [x for x in range(n_train+1,n_test+n_train)]


    save_path = "training_data/value_cnn"
    main(n_train,n_test,save_path,train_seeds,test_seeds,num_reward_variants=3)


    






