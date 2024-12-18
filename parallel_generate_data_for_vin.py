import os
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
import h5py
from copy import deepcopy
import multiprocessing

# =============== Your original helper functions ===============

def get_full_trajectory(n, rewards, obstacles_map, neighbors, start):
    agent_position = deepcopy(start)
    steps = 0
    max_steps = 1000
    path = [agent_position]
    reward_map_list = []

    gamma = 0.9  # discount factor

    while np.any(rewards) and steps < max_steps:
        rewards[agent_position[0], agent_position[1]] = 0
        reward_map_list.append(rewards.copy().reshape(1, n, n))

        V = value_iteration(n, rewards, obstacles_map, gamma, neighbors)
        policy = extract_policy(V, obstacles_map, neighbors, n)
        
        # Move agent
        next_position = tuple(int(i) for i in policy[agent_position])
        agent_position = next_position
        path.append(agent_position)
        steps += 1

    obstacles_map = np.where(obstacles_map, 1, 0).reshape(1, n, n)
    reward_map_list = [np.concatenate((img, obstacles_map), axis=0) for img in reward_map_list]

    return np.array(path), np.array(reward_map_list)

def extract_action(traj):
    """Given a trajectory, extract the actions that were taken (0,1,2,3)."""
    actions = []
    action_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
    state_diff = np.diff(traj, axis=0)
    for diff in state_diff:
        actions.append(action_map[tuple(diff)])
    return np.array(actions)

def sample_trajectories(num_trajectories, reward, obstacle_map, n):
    """Generate multiple trajectories for a given reward and obstacle map."""
    states_xy = []
    neighbors = precompute_next_states(n, obstacle_map)
    for _ in range(num_trajectories):
        start, goal = pick_start_and_goal(reward, obstacle_map)
        path = get_vi_path(n, reward, obstacle_map, neighbors, start, goal)
        states_xy.append(np.array(path))
    return states_xy

# =============== Parallel data-generation logic ===============

def generate_data_for_seed(seed, num_reward_variants, n, min_obstacles, max_obstacles, 
                           obstacle_type="block", square_size=25, 
                           num_reward_blocks=(2,20), reward_square_size=(2,20),
                           obstacle_cluster_prob=0.4, obstacle_square_sizes=(3,20)):
    """
    Generate training samples for a single seed. Returns X, S1, S2, Labels arrays.
    """
    X_list = []
    S1_list = []
    S2_list = []
    Labels_list = []

    # Generate initial obstacle map & reward
    reward, obstacle_map = init_random_reachable_map(
        n=n,
        config="block",
        num_blocks=5,
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
        obstacle_type=obstacle_type,
        square_size=square_size,
        obstacle_map=None,
        seed=seed,
        num_reward_blocks=num_reward_blocks,
        reward_square_size=reward_square_size,
        obstacle_cluster_prob=obstacle_cluster_prob,
        obstacle_square_sizes=obstacle_square_sizes
    )
    neighbors = precompute_next_states(n, obstacle_map)

    for variant in range(num_reward_variants):
        # Generate a new reward map for each variant
        reward_variant, obstacle_map = init_random_reachable_map(
            n=n,
            config="block",
            num_blocks=5,
            min_obstacles=min_obstacles,
            max_obstacles=max_obstacles,
            obstacle_type=obstacle_type,
            square_size=square_size,
            obstacle_map=obstacle_map,  # re-use obstacle map or regenerate
            seed=seed,
            num_reward_blocks=num_reward_blocks,
            reward_square_size=reward_square_size,
            obstacle_cluster_prob=obstacle_cluster_prob,
            obstacle_square_sizes=obstacle_square_sizes
        )

        if np.sum(reward_variant) == 0:
            # If no positive rewards, skip
            continue
        
        states_xy, reward_list = get_full_trajectory(n, reward_variant.copy(), obstacle_map, neighbors, start=(0, 0))
        if len(reward_list) == 0:
            continue
        
        # Check for any (0,0) moves that might indicate a stuck agent
        state_diff = np.diff(states_xy, axis=0)
        if any(tuple(sd) == (0,0) for sd in state_diff):
            continue

        actions = extract_action(states_xy)
        states_xy = states_xy[:-1]  # remove last state
        if reward_list.shape[0] != states_xy.shape[0]:
            continue

        # Prepare data
        S1_cur = np.expand_dims(states_xy[:, 0], axis=1)  # x coords
        S2_cur = np.expand_dims(states_xy[:, 1], axis=1)  # y coords
        Labels_cur = np.expand_dims(actions, axis=1)

        X_list.append(reward_list)
        S1_list.append(S1_cur)
        S2_list.append(S2_cur)
        Labels_list.append(Labels_cur)

    if len(X_list) == 0:
        return (np.empty((0,2,n,n)), 
                np.empty((0,1)), 
                np.empty((0,1)), 
                np.empty((0,1)))

    X_conc = np.concatenate(X_list, axis=0)
    S1_conc = np.concatenate(S1_list, axis=0)
    S2_conc = np.concatenate(S2_list, axis=0)
    Labels_conc = np.concatenate(Labels_list, axis=0)
    
    return X_conc, S1_conc, S2_conc, Labels_conc

def vin_data_parallel(n_seeds, seeds, num_reward_variants=7, n=50, 
                      min_obstacles=10, max_obstacles=20):
    """
    Generate data in parallel for a list of seeds. Returns X, S1, S2, Labels (concatenated).
    """
    pool_inputs = []
    for seed in seeds:
        pool_inputs.append((seed, num_reward_variants, n, min_obstacles, max_obstacles))

    num_processes = min(len(seeds), os.cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(generate_data_for_seed, pool_inputs)

    X_list, S1_list, S2_list, Labels_list = [], [], [], []
    for (X_cur, S1_cur, S2_cur, L_cur) in results:
        if X_cur.shape[0] > 0:
            X_list.append(X_cur)
            S1_list.append(S1_cur)
            S2_list.append(S2_cur)
            Labels_list.append(L_cur)
    
    if len(X_list) == 0:
        return (np.empty((0,2,n,n)), 
                np.empty((0,1)), 
                np.empty((0,1)), 
                np.empty((0,1)))

    X = np.concatenate(X_list, axis=0)
    S1 = np.concatenate(S1_list, axis=0)
    S2 = np.concatenate(S2_list, axis=0)
    Labels = np.concatenate(Labels_list, axis=0)

    print(f"Shapes from vin_data_parallel: X={X.shape}, S1={S1.shape}, S2={S2.shape}, Labels={Labels.shape}")
    return X, S1, S2, Labels

# =============== Sharding Helper ===============

def save_shards_hdf5(X, S1, S2, Labels, output_dir, prefix="data", shard_size=10000):
    """
    Splits (X, S1, S2, Labels) into multiple shards of 'shard_size' each,
    and saves each shard to an HDF5 file in 'output_dir'.
    
    - X shape: (N, 2, n, n)
    - S1 shape: (N, 1)
    - S2 shape: (N, 1)
    - Labels shape: (N, 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    total_samples = X.shape[0]
    shard_count = (total_samples + shard_size - 1) // shard_size  # ceiling division

    print(f"Saving {total_samples} samples into {shard_count} shards (size={shard_size} each).")

    start_idx = 0
    for shard_idx in range(shard_count):
        end_idx = min(start_idx + shard_size, total_samples)
        shard_filename = os.path.join(output_dir, f"{prefix}_shard_{shard_idx}.h5")

        with h5py.File(shard_filename, 'w') as hf:
            hf.create_dataset("X", data=X[start_idx:end_idx], compression="gzip")
            hf.create_dataset("S1", data=S1[start_idx:end_idx], compression="gzip")
            hf.create_dataset("S2", data=S2[start_idx:end_idx], compression="gzip")
            hf.create_dataset("Labels", data=Labels[start_idx:end_idx], compression="gzip")

        print(f"Shard {shard_idx} saved: samples [{start_idx}, {end_idx}). File: {shard_filename}")
        start_idx = end_idx


# =============== Main script ===============

def main(n_train, n_test, output_dir, train_seeds, test_seeds, num_reward_variants=7, n=50, shard_size=10000,n_workers=4):
    """
    Generate training/test data in parallel, then save them in multiple shards of HDF5 files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating training data (parallelized)")
    X_train, S1_train, S2_train, Labels_train = vin_data_parallel(
        n_train, train_seeds, num_reward_variants=num_reward_variants, n=n
    )
    
    print("Generating test data (parallelized)")
    X_test, S1_test, S2_test, Labels_test = vin_data_parallel(
        n_test, test_seeds, num_reward_variants=num_reward_variants, n=n
    )

    # Save training shards
    save_shards_hdf5(
        X_train, S1_train, S2_train, Labels_train,
        output_dir=output_dir,
        prefix="train",
        shard_size=shard_size
    )

    # Save test shards
    save_shards_hdf5(
        X_test, S1_test, S2_test, Labels_test,
        output_dir=output_dir,
        prefix="test",
        shard_size=shard_size
    )

    print("All shards saved successfully.")

if __name__ == "__main__":
    # Example usage
    n = 50
    n_train = 2
    n_test = 2
    
    train_seeds = list(range(n_train))           # e.g., [0,1]
    test_seeds = list(range(n_train, n_train+n_test))  # e.g., [2,3]

    output_dir = "training_data/"
    shard_size = 1000  # Adjust as needed, e.g. 10k or 100k for larger scale
    n_workers = 4

    main(
        n_train=n_train,
        n_test=n_test,
        output_dir=output_dir,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        num_reward_variants=2,  # how many variants per seed
        n=n,
        shard_size=shard_size,
        n_workers=n_workers
    )
