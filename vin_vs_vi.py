import numpy as np
import matplotlib.pyplot as plt
from utils import *
from heuristics import *
from nn_training import *
from fo_solver import *
import pickle
import torch
from eval import *
from dl_models import *
from get_training_data import * 
from pytorch_value_iteration_networks.model import *
import time
import pandas as pd
from types import SimpleNamespace
import seaborn as sns
from queue import Queue

from puct import PUCT
from modified_gridenv import ModifiedGridEnvironment
from tabluar_puct import tabPUCT

# Ensure the plot style is set
sns.set(style="whitegrid")

# Initialize a DataFrame to keep track of the statistics
results_df = pd.DataFrame(columns=['seed', 'algorithm', 'mean_inference_time', 'total_steps', 'successful','path'])

def is_reachable(obstacles_map, start, goal):
    n = obstacles_map.shape[0]
    visited = np.zeros_like(obstacles_map, dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    q = Queue()
    q.put(start)
    visited[start[0], start[1]] = True

    while not q.empty():
        current = q.get()
        if current == goal:
            return True
        for d in directions:
            new_pos = (current[0] + d[0], current[1] + d[1])
            if 0 <= new_pos[0] < n and 0 <= new_pos[1] < n and not visited[new_pos[0], new_pos[1]] and not obstacles_map[new_pos[0], new_pos[1]]:
                visited[new_pos[0], new_pos[1]] = True
                q.put(new_pos)

    return False

def plot_results(results_df):
    # Bar Plot of Mean Inference Time per Algorithm
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithm', y='mean_inference_time', data=results_df)
    plt.title('Mean Inference Time per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Inference Time (seconds)')
    plt.show()

    # Box Plot for Inference Time Distribution across seeds
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='algorithm', y='mean_inference_time', data=results_df)
    plt.title('Inference Time Distribution per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Inference Time (seconds)')
    plt.show()

    # Bar Plot for Total Steps per Algorithm
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithm', y='total_steps', data=results_df)
    plt.title('Total Steps per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Total Steps')
    plt.show()

    # Success Rate Plot
    plt.figure(figsize=(10, 6))
    success_rates = results_df.groupby('algorithm')['successful'].mean().reset_index()
    sns.barplot(x='algorithm', y='successful', data=success_rates)
    plt.title('Success Rate per Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)  # Ensure the y-axis is between 0 and 1
    plt.show()


def get_vin_path(vin, n, obstacle_map, rewards, start, goal):
    actions = {0: (0, -1),
               1: (1, 0),
               2: (0, 1),
               3: (-1, 0)}  # up, right, down, left

    checker = LiveLockChecker(counter=0, last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 1000
    step = 0
    inference_time = []

    while agent_pos != goal and step < max_step:
        rewards[agent_pos[0], agent_pos[1]] = 0
        input = reformat_input(rewards, obstacle_map)
        input = input.unsqueeze(0)
        assert input.shape == (1, 2, n, n)

        start_time = time.time()

        logits, _ ,_= vin(input.to(device), torch.tensor(agent_pos[0]).to(device), torch.tensor(agent_pos[1]).to(device), 50)

        inference_duration = time.time() - start_time
        inference_time.append(inference_duration)

        pred = torch.argmax(logits).item()
        action = actions[pred]

        new_pos = tuple([agent_pos[0] + action[0], agent_pos[1] + action[1]])

        checker.update(agent_pos, new_pos)
        if checker.check(agent_pos, new_pos):
            print("Live Lock Detected in VIN")
            return path, False, np.mean(inference_time), len(path), "Live Lock Detected"
        
        if obstacle_map[new_pos[0], new_pos[1]]:
            print("Agent moved into an obstacle")
            return path, False, np.mean(inference_time), len(path), "collision"

        agent_pos = new_pos
        path.append(agent_pos)
        step += 1

    success = agent_pos == goal

    reason = "Goal Reached" if success else "Max Steps Reached"

    return path, success, np.mean(inference_time), len(path), reason 


def get_vi_path(n, rewards, obstacle_map, neighbors, start, goal):
    # Placeholder for your VI pathfinding algorithm implementation
    # Track inference time, steps, and success similarly as in get_vin_path
    agent_pos = deepcopy(start)
    checker = LiveLockChecker(counter=0, last_visited={})
    path = [agent_pos]
    max_step = 1000
    step = 0
    inference_time = []

    while agent_pos != goal and step < max_step:
        rewards[agent_pos[0], agent_pos[1]] = 0
        start_time = time.time()
        # Call your VI path function here
        V = value_iteration(n, rewards.copy(), obstacle_map, gamma=0.9, neighbors=neighbors)  
        end_time = time.time() - start_time
        inference_time.append(end_time)

        policy = extract_policy(V, obstacle_map, neighbors, n)
        next_pos = tuple(int(i) for i in policy[agent_pos])

        checker.update(agent_pos, next_pos)
        if checker.check(agent_pos, next_pos):
            print("Live Lock Detected in VI")
            return path, False, np.mean(inference_time), len(path), "Live Lock Detected"
        

        if obstacle_map[next_pos[0], next_pos[1]]:
            print("Agent moved into an obstacle")
            return path, False, np.mean(inference_time), len(path), "collision"


        agent_pos = next_pos
        path.append(agent_pos)
        step += 1

    success = agent_pos == goal
    reason = "Goal Reached" if success else "Max Steps Reached"

    return path, success, np.mean(inference_time), len(path), reason


def get_vin_path_with_value_hueristic(vin, n, obstacle_map,rewards, start,goal,k = 16):


    actions = {0:(0,-1),
               1:(1,0),
               2:(0,1),
               3:(-1,0)} # up, right, down , left

    checker = LiveLockChecker(counter=0,last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 150
    step = 0

    infrence_time = []
    total_reward = 0
    total_reward_list =[]

    while agent_pos!=goal and step<max_step:
        total_reward+=rewards[agent_pos[0],agent_pos[1]]
        total_reward_list.append(total_reward)
        rewards[agent_pos[0],agent_pos[1]] = 0
        input = reformat_input(rewards,obstacle_map)
        input = input.unsqueeze(0)
        assert input.shape == (1,2,n,n) 

        start = time.time()

        logits,_,_ = vin(input,torch.tensor(agent_pos[0]),torch.tensor(agent_pos[1]),k)

        end = time.time() - start

        infrence_time.append(end)

        pred = torch.argmax(logits).item()

        action = actions[pred]

        new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])

        checker.update(agent_pos,new_pos)
        if checker.check(agent_pos,new_pos) or obstacle_map[new_pos]:
            print("Live Lock or collision detected, swithcing to A* heuristic")

            start = time.time()
            heuristic_path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)
            end = time.time() - start
            infrence_time.append(end)

            if not heuristic_path:
                print("Heuristic failed")
                return path, False, np.mean(infrence_time), len(path), "Heuristic Failed"

            for i in range(1,len(heuristic_path)):
                next_pos = heuristic_path[i]
                agent_pos = next_pos
                path.append(agent_pos)
                if agent_pos == goal:
                    break
            
            continue

            # new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])
            # checker.update(agent_pos,new_pos)

        
        agent_pos = new_pos

        path.append(agent_pos)


    if agent_pos == goal:
        return path, True, np.mean(infrence_time), len(path), "Goal Reached", 
    else:
        return path, False, np.mean(infrence_time), len(path), "Max Steps Reached",



def get_puct_path(vin, config, obstacle_map, rewards, start, goal, k=50, max_steps=10000, gamma=1, c_puct=1.44):
    actions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # up, right, down, left
    checker = LiveLockChecker(counter=0, last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    inference_times = []
    step = 0
    success = False
    reason = "Max Steps Reached"  # Default reason if no other termination condition met
    env = ModifiedGridEnvironment(config, rewards.copy(), obstacle_map, agent_pos, goal, living_reward=None, max_steps=1000)
    puct = PUCT(vin, env, agent_pos, gamma=gamma, c_puct=c_puct,alpha=0.15,epsilon=0)
    # puct = tabPUCT(vin, env, agent_pos, gamma=gamma, c_puct=c_puct,k=50,alpha=0.15,epsilon=0)
    # puct = 
    while agent_pos != goal and step < max_steps:
        rewards[agent_pos[0], agent_pos[1]] = 0
        # env = GridEnvironment(config, rewards.copy(), obstacle_map, agent_pos, goal, max_steps=1000,living_reward=-0.01)   
        
        # Timing the inference
        start_time = time.time()
        action, new_position = puct.search(agent_pos,rewards.copy(),num_simulations=150)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        puct.update_root(action)
        
        # Update for potential live-lock
        checker.update(agent_pos, new_position)
        if checker.check(agent_pos, new_position):
            reason = "Live Lock Detected in PUCT"
            print(reason)
            break

        if obstacle_map[new_position[0], new_position[1]]:
            reason = "Agent moved into an obstacle, PUCT"
            print(reason)
            return path, False, np.mean(inference_times), len(path), reason

        path.append(new_position)
        agent_pos = new_position
        step += 1

    # Check if the goal was reached
    success = agent_pos == goal
    if success:
        reason = "Goal Reached"

    # Calculate mean and total inference time
    mean_inference_time = np.mean(inference_times)

    return path, success, mean_inference_time, len(path), reason

def compare_paths(paths, rewards, obstacles_map, target_location,seed=None,titles=None):
    num_paths = len(paths)
    fig, ax = plt.subplots(1, num_paths, figsize=(14, 10))  # Adjust figure size based on number of value functions

    for i in range(num_paths):
        ax_i = ax[i] if num_paths > 1 else ax  # Handle single plot case
        display_matrix = np.copy(rewards)
        display_matrix[obstacles_map] = np.nan  # Set obstacles to NaN for black color

        im = ax_i.imshow(display_matrix, cmap='viridis', origin='upper')
        ax_i.plot(target_location[1], target_location[0], 'ro')
        ax_i.plot(0, 0, 'wo')

        path = paths[i]
        for j in range(len(path) - 1):
            ax_i.annotate('', xy=(path[j+1][1], path[j+1][0]), xytext=(path[j][1], path[j][0]),
                          arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'))
            
        if titles:
            ax_i.set_title(titles[i])
        else: 
            ax_i.set_title(f"Path {i + 1}")
        # fig.colorbar(im, ax=ax_i)

        ax_i.invert_yaxis()
        ax_i.grid(True)

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.04)
    if seed is not None:
        plt.suptitle(f"Comparison of Paths (Seed: {seed})")


    plt.savefig("images/puct" + f"{seed}"+"vin_comparison.png", format='png')  # Save the figure as a PNG file
    # plt.show()



if __name__ == "__main__":
    import argparse

    successful_paths = []
    failed_paths = []
    n = 20
    seeds = np.random.randint(6100, 20000, 100)
    # seeds = [9651]

    pnet = PNetResNet(2, 128, 128, 8)
    pnet_path = "model_weights/pnet_resnet_2.pth"

    cae_net = ContractiveAutoEncoder()
    cae_net_path = "model_weights/CAE_1.pth"

    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True, map_location=device)
 

    parser = argparse.ArgumentParser()

    ### Gridworld parameters

    parser.add_argument('--n', type=int, default=20, help='Grid size')
    parser.add_argument('--obstacle_shape', type=str, default="block", help='Shape of obstacles')
    parser.add_argument('--num_obstacles', type=int, default=5, help='Number of obstacles')
    parser.add_argument('--min_obstacles', type=int, default=2, help='Minimum obstacles')
    parser.add_argument('--max_obstacles', type=int, default=10, help='Maximum obstacles')
    parser.add_argument('--obstacle_type', type=str, default="block", help='Type of obstacles')
    parser.add_argument('--square_size', type=int, default=25, help='Size of the grid square')
    parser.add_argument('--obstacle_map', default=None, help='Initial obstacle map')
    #parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_reward_blocks', type=tuple, default=(2, 5), help='Range of reward blocks')
    parser.add_argument('--reward_square_size', type=tuple, default=(4, 6), help='Size of reward squares')
    parser.add_argument('--obstacle_cluster_prob', type=float, default=0.3, help='Probability of obstacle clustering')
    parser.add_argument('--obstacle_square_sizes', type=tuple, default=(3, 8), help='Range of obstacle square sizes')
    parser.add_argument('--living_reward', type=float, default=-0.1, help='Living reward for each step')

      
    # VIN-specific parameters
    parser.add_argument('--k', type=int, default=50, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_sz', type=int, default=1, help='Batch size')

    config = parser.parse_args()



    vin = VIN(config)
    vin.load_state_dict(vin_weights)

    vin.to(device)
    vin.eval()

    for seed in seeds:
        print(f"Running seed {seed}")
        rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, 2, 20, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
        if np.sum(rewards) == 0:
            continue
        neighbors = precompute_next_states(n, obstacles_map)
        start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)

        if obstacles_map[start[0], start[1]]:
            print(f"Skipping seed {seed}: start position is on an obstacle")
            continue

        # Check if the goal is reachable from the start
        if not is_reachable(obstacles_map, start, goal):
            print(f"Skipping seed {seed}: goal is not reachable from start")
            continue


        # Get VI path
        path_vi, success_vi, mean_inf_time_vi, total_steps_vi,reason = get_vi_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)

        # Append VI results to the DataFrame
        new_row_vi = pd.DataFrame({
            'seed': [seed],
            'algorithm': ['VI'],
            'mean_inference_time': [mean_inf_time_vi],
            'total_steps': [total_steps_vi],
            'successful': [success_vi],
            'path': [path_vi],
            'stop_reason': [reason]
        })
        results_df = pd.concat([results_df, new_row_vi], ignore_index=True)

        # Get VIN path
        path_vin, success_vin, mean_inf_time_vin, total_steps_vin,reason = get_vin_path(vin, 20, obstacles_map, rewards.copy(), start, goal)

        # Append VIN results to the DataFrame
        new_row_vin = pd.DataFrame({
            'seed': [seed],
            'algorithm': ['VIN'],
            'mean_inference_time': [mean_inf_time_vin],
            'total_steps': [total_steps_vin],
            'successful': [success_vin],
            'path_vin': [path_vin],
            'stop_reason': [reason]
        })
        results_df = pd.concat([results_df, new_row_vin], ignore_index=True)

        path_puct, success_puct, mean_inf_time_puct, total_steps_puct,reason = get_puct_path(vin,config,obstacles_map,rewards.copy(),start,goal,gamma=0.99,c_puct=1.4)

        new_row_puct = pd.DataFrame({
            'seed': [seed],
            'algorithm': ['PUCT'],
            'mean_inference_time': [mean_inf_time_puct],
            'total_steps': [total_steps_puct],
            'successful': [success_puct],
            'path_puct': [path_puct],
            'stop_reason': [reason]
        })

        results_df = pd.concat([results_df, new_row_puct], ignore_index=True)



    # Save the results to a CSV file if needed
    results_df.to_csv('path_results_3.csv', index=False)

    # Print the DataFrame for quick view

    # Plot results
    plot_results(results_df)

    # successful_df = results_df[results_df['successful'] == True]
    # failed_df = results_df[results_df['successful'] == False]

    # # Sample 3 successful and 3 failed paths
    # successful_sample = successful_df.sample(min(3, len(successful_df)))
    # failed_sample = failed_df.sample(min(3, len(failed_df)))
    # print("Visualizing Successful Paths")
    # compare_paths(successful_sample['path'].tolist(), successful_sample.iloc[0]['rewards'], successful_sample.iloc[0]['obstacles_map'], successful_sample.iloc[0]['goal'], titles=["Success" for _ in range(len(successful_sample))])

    # # Visualize 3 failed paths
    # print("Visualizing Failed Paths")
    # compare_paths(failed_sample['path'].tolist(), failed_sample.iloc[0]['rewards'], failed_sample.iloc[0]['obstacles_map'], failed_sample.iloc[0]['goal'], titles=["Failed" for _ in range(len(failed_sample))])
