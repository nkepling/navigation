import pickle
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from eval import get_vi_path
from nn_training import * 
import argparse
from tqdm import tqdm
import fo_solver

from multiprocessing import Pool


# define experiment configuration
# random_map = True

def get_full_trajectory(n, config,rewards, obstacles_map, neighbors, start):
    agent_position = deepcopy(start)
    steps = 0
    max_steps = config["max_steps"]
    gamma = config["gamma"]

    path = [agent_position]
    reward_map_list = []

    while np.any(rewards) and steps < max_steps:
        # Zero out the reward at the current agent's position
        rewards[agent_position[0], agent_position[1]] = 0
        
        # Append a copy of the rewards map at this step to avoid mutating it in future iterations
        reward_map_list.append(rewards.copy().reshape(1, n, n))
        
        # Run value iteration to get the value function and policy
        
        V = value_iteration(n, rewards, obstacles_map, gamma, neighbors)
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

def single_seed_vin_data(seed,config):
    n = config["n"]
    rewards_config=config["rewards_config"]
    min_obstacles=config["min_obstacles"]
    max_obstacles=config["max_obstacles"]
    obstacle_type=config["obstacle_type"]

    num_reward_blocks=config["num_reward_blocks"]
    reward_square_size=config["reward_square_size"]
    obstacle_cluster_prob=config["obstacle_cluster_prob"]
    obstacle_square_sizes=config["obstacle_square_sizes"]
    num_reward_variants = config["num_reward_variants"]
    obstacle_map = config["obstacle_map"]

    X = []
    S1 = []
    S2 = []
    Labels = []

    reward, obstacle_map = init_random_reachable_map(n = n,
                                                    rewards_config=rewards_config,
                                                    min_obstacles=min_obstacles,
                                                    max_obstacles=max_obstacles,
                                                    obstacle_type=obstacle_type,
                                                    obstacle_map=obstacle_map,
                                                    seed=seed,
                                                    num_reward_blocks=num_reward_blocks,
                                                    reward_square_size=reward_square_size,
                                                    obstacle_cluster_prob=obstacle_cluster_prob,
                                                    obstacle_square_sizes=obstacle_square_sizes
                                                    )
    
    neighbors = precompute_next_states(n , obstacle_map) 

    # For each map configuration, create `num_reward_variants` different reward distributions
    for variant in range(num_reward_variants):
    
        reward, obstacle_map = init_random_reachable_map(n = n,
                                                        rewards_config=rewards_config,
                                                        min_obstacles=min_obstacles,
                                                        max_obstacles=max_obstacles,
                                                        obstacle_type=obstacle_type,
                                                        obstacle_map=obstacle_map,
                                                        seed=seed,
                                                        num_reward_blocks=num_reward_blocks,
                                                        reward_square_size=reward_square_size,
                                                        obstacle_cluster_prob=obstacle_cluster_prob,
                                                        obstacle_square_sizes=obstacle_square_sizes
                                                        )
        
        if np.sum(reward) == 0:
            print("No reward skipping")
            continue

        # Get trajectories and reward maps
        states_xy, reward_list = get_full_trajectory(n, config, reward.copy(), obstacle_map, neighbors, start=(0, 0))
        
        # Skip empty reward lists (i.e., no trajectory or no rewards)
        if len(reward_list) == 0:
            print(f"Skipping empty trajectory for seed {seed}, variant {variant}")
            continue
        
        skip_variant = False

        state_diff = np.diff(states_xy, axis=0)  # Calculate state transitions
        for i in range(len(state_diff)):
            diff = tuple(state_diff[i])
            if diff == (0, 0):
                print(f"Skipping (0, 0) movement at step {i}")
                skip_variant = True
                break
        
        if skip_variant:
            continue 

        actions = extract_action(states_xy)  # Extract actions from the trajectory
        states_xy = states_xy[:-1]  # Remove last state as it corresponds to the final state
        assert reward_list.shape == (len(states_xy), 2, n, n), f"reward_list shape {reward_list.shape}"

        # Prepare the data
        S1_cur = np.expand_dims(states_xy[:, 0], axis=1)  # x coordinates
        S2_cur = np.expand_dims(states_xy[:, 1], axis=1)  # y coordinates
        Labels_cur = np.expand_dims(actions, axis=1)  # actions taken

        # Append the data to lists
        X.append(reward_list)
        S1.append(S1_cur)
        S2.append(S2_cur)
        Labels.append(Labels_cur)

    # Concatenate all data
    X = np.concatenate(X, axis=0)
    S1 = np.concatenate(S1, axis=0)
    S2 = np.concatenate(S2, axis=0)
    Labels = np.concatenate(Labels, axis=0)

    return X, S1, S2, Labels

def data_gen_wrapper(args):
    return single_seed_vin_data(*args)

def vin_data(seeds,config):

    num_workers = config["num_workers"]

    results = []
    with Pool(num_workers) as pool, tqdm(total=len(seeds)) as pbar:
        for result in pool.imap_unordered(data_gen_wrapper,[(seed,config) for seed in seeds]):
            results.append(result)
            pbar.update(1)

    print(len(results))

    X_list, S1_list, S2_list, Labels_list = zip(*results)

    X = np.concatenate(X_list, axis=0)
    S1 = np.concatenate(S1_list, axis=0)
    S2 = np.concatenate(S2_list, axis=0)
    Labels = np.concatenate(Labels_list, axis=0)
    return X, S1, S2, Labels

def main(save_path, train_seeds,test_seeds,config):
    print("Generating training data for VIN model")
    X, S1, S2, Labels = vin_data(train_seeds,config)  # 10 reward variants per map config
    print("Generating test data")
    X_test, S1_test, S2_test, Labels_test = vin_data(test_seeds,config)  # 10 reward variants per map config
    np.savez_compressed(save_path, X, S1, S2, Labels, X_test, S1_test, S2_test, Labels_test)

    print(f"Saved data to {save_path}")
    return X, S1, S2, Labels

if __name__ == "__main__":
    import argparse
    from experiments.experiment_setup import read_config

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str,required=True)
    args = parser.parse_args()
    config = read_config(args.config_path)

    n_train = config["n_train"]
    n_test = config["n_test"]
    num_reward_variants = config["num_reward_variants"]
    #n_rewards = 3

    dataset_name = config["dataset_name"]

    train_seeds = [x for x in range(n_train)]
    test_seeds = [x for x in range(n_train+1,n_test+n_train)]

    os.makedirs("training_data",exist_ok=True)

    save_path = "training_data/" + dataset_name + ".npz"

    
    X,S1,S2,Labels = main(save_path,train_seeds,test_seeds,config)
    
    print("X ",X.shape)
    print("S1 ",S1.shape)
    print("S2 ",S2.shape)
    print("Labels ",Labels.shape)







