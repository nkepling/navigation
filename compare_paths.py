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
from gridworld_env import GridEnvironment
from modified_gridenv import ModifiedGridEnvironment
from puct import *


def get_puct_path(vin,config,obstacle_map, rewards, start, goal, k=50, max_steps=10000, gamma=1, c_puct=1.44):
    actions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # up, right, down, left
    checker = LiveLockChecker(counter=0, last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    inference_times = []
    step = 0

    while agent_pos != goal and step < max_steps:
        rewards[agent_pos[0], agent_pos[1]] = 0
        env = GridEnvironment(config, rewards.copy(), obstacle_map, agent_pos, goal,max_steps=1000) # this is wrong.. .
        puct = PUCT(vin,env, agent_pos,gamma=gamma,c_puct=c_puct)
        # Timing the inference
        start_time = time.time()
        action, new_position = puct.search(num_simulations=100)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Update for potential live-lock
        checker.update(agent_pos, new_position)
        if checker.check(agent_pos, new_position):
            print("Live Lock Detected")
            break

        path.append(new_position)
        agent_pos = new_position
        step += 1
        # print(f"Step {step}: Agent moved to {new_position}")

    mean_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)

    print("Mean inference time:", mean_inference_time)
    print("Total time to reach goal:", total_inference_time)

    path = [(p, "PUCT") for p in path]
    return path, mean_inference_time, total_inference_time

def get_vi_path_with_fov(n, rewards, obstacles_map, neighbors, start, goal):
    agent_position = deepcopy(start)
    steps = 0
    path = [agent_position]
    reward_list = []
    time_list = []
    checker = LiveLockChecker(last_visited={}, counter=0)
    rewards[agent_position[0], agent_position[1]] = 0
    goal_in_fov = False

    

    while agent_position!=goal or not goal_in_fov:
        visualize_rewards(rewards, obstacles_map, agent_position, goal)

        start_time = time.time()
        Viter = value_iteration_with_extened_fov(n,rewards,obstacles_map,gamma,neighbors)
        policy = extract_policy(Viter, obstacles_map,neighbors,n=n)
        end_time = time.time()
        print(f"Time for VI is {end_time - start_time}")
        time_list.append(end_time - start_time)
        next_position = tuple(int(i) for i in policy[agent_position])
        checker.update(agent_position, next_position)
        if checker.check(agent_position, next_position):
            print("Live Lock Detected")
            break

        fov = get_fov(agent_position,next_position,obstacles_map,obstacles_map.shape)

        if goal in fov:
            print("found goal in fov")
            goal_in_fov = True

        fov = np.array(fov)
        rewards[fov[:,0],fov[:,1]] = 0

        agent_position = next_position


        path.append(agent_position)
        reward_list.append(rewards)
        steps += 1

    path_list = [(x,"fov") for x in path]
    print(f"mean time for VI is {np.mean(time_list)}")
    print("num steps ",steps)    

    print("found goal", agent_position == goal)
    return path_list, np.mean(time_list),steps

def density_aware_vi(n, rewards, obstacles_map, neighbors, start, goal):
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 10000
    step = 0

    while agent_pos != goal and step < max_step:
        rewards[agent_pos] = 0
        density_reward_map = create_density_based_reward_map(rewards,0.7,1)
        V = value_iteration(n, density_reward_map, obstacles_map, gamma=0.9, neighbors=neighbors)
        policy = extract_policy(V, obstacles_map, neighbors, n)
        next_pos = tuple(int(i) for i in policy[agent_pos])
        agent_pos = next_pos
        path.append(agent_pos)
        step += 1
    

    return [(p,"Density_Aware_VI") for p in path]

def get_vin_path(vin, n, obstacle_map,rewards, start,goal,k = 50):


    actions = {0:(0,-1),
               1:(1,0),
               2:(0,1),
               3:(-1,0)} # up, right, down,left 

    checker = LiveLockChecker(counter=0,last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 50
    step = 0

    infrence_time = []

    while agent_pos!=goal and step<max_step:
        rewards[agent_pos[0],agent_pos[1]] = 0
        input = reformat_input(rewards,obstacle_map)
        input = input.unsqueeze(0)
        assert input.shape == (1,2,n,n) 

        start = time.time()

        logits,preds,v = vin(input,torch.tensor(agent_pos[0]),torch.tensor(agent_pos[1]),k)

        end = time.time() - start

        infrence_time.append(end)

        pred = torch.argmax(logits).item()

        action = actions[pred]

        new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])

        checker.update(agent_pos,new_pos)
        if checker.check(agent_pos,new_pos):
            print("Live Lock Detected")
            print(np.mean(infrence_time))
            break
            # path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)


            # new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])
            # checker.update(agent_pos,new_pos)

        
        agent_pos = new_pos

        path.append(agent_pos)

    path = [(p,"VIN") for p in path]
    print("mean infrence time" ,np.mean(infrence_time))
    print("total time to reach goal", np.sum(infrence_time))
    return path

def get_vin_path_with_value(vin, n, obstacle_map,rewards, start,goal,k = 16):


    if torch.cuda.is_available():
        device = torch.device('cuda')


    else:
        device = torch.device('cpu')



    actions = {0:(0,-1),
               1:(1,0),
               2:(0,1),
               3:(-1,0)} # up, right, down , left

    #checker = LiveLockChecker(counter=0,last_visited={})
    agent_pos = deepcopy(start)
    path = [(agent_pos,"VIN")]
    max_step = 500
    step = 0

    infrence_time = []

    while agent_pos!=goal and step<max_step:
        rewards[agent_pos[0],agent_pos[1]] = 0
        input = reformat_input(rewards,obstacle_map)
        input = input.unsqueeze(0)
        assert input.shape == (1,2,n,n) 

        start = time.time()

        logits, _, _ = vin(
                        input.to(device).type(torch.float32),  # Ensure input is float32 and on the device
                        torch.tensor(agent_pos[0], dtype=torch.float32).to(device),  # Explicitly set dtype to float32
                        torch.tensor(agent_pos[1], dtype=torch.float32).to(device),  # Explicitly set dtype to float32
                        k)

        end = time.time() - start

        infrence_time.append(end)

        pred = torch.argmax(logits).item()

        action = actions[pred]

        new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])

        # checker.update(agent_pos,new_pos)
        # if checker.check(agent_pos,new_pos):
        #     print("Live Lock Detected")
        #     print(np.mean(infrence_time))
        #     break
            # path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)


            # new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])
            # checker.update(agent_pos,new_pos)

        
        agent_pos = new_pos
        step+=1

        path.append((agent_pos,"VIN"))  


    print("mean infrence time" ,np.mean(infrence_time))
    print("total time to reach goal", np.sum(infrence_time))

    return path

def get_p_net_path(cae_net,cae_net_path, pnet,pnet_path,obstacles_map,rewards,start,goal):

    action_map = {0:(-1,0),
                  1:(0,1),
                  2:(1,0),
                  3:(0,-1)}

    agent_pos = deepcopy(start)
    checker = LiveLockChecker(counter=0,last_visited={})
    pathlist = []
    pathlist.append(agent_pos)
    
    cae_net.load_state_dict(torch.load(cae_net_path,weights_only=True))
    pnet.load_state_dict(torch.load(pnet_path,weights_only=True))
    max_step = 50
    step = 0
    while agent_pos!= goal and step<max_step:
        input = reformat_input_for_mpnet(obstacles_map,rewards)
        _,latent = cae_net(input.float())
        
        coord = torch.tensor(agent_pos).unsqueeze(0)
        action_logits = pnet(coord,latent)

        action = torch.argmax(action_logits,dim=1).item()


        dir = action_map[action]

        next_pos = (agent_pos[0] + dir[0],agent_pos[1]+dir[1])

        checker.update(agent_pos,next_pos)
        if checker.check(agent_pos,next_pos):
            print("Live lock detected")

            return pathlist

        agent_pos = next_pos

        pathlist.append(agent_pos)  
        step+=1
    
    return pathlist

def get_heuristic_path(n, rewards, obstacles_map, neighbors, start, goal):
    agent_pos = deepcopy(start)
    checker = LiveLockChecker(counter=0, last_visited={})
    model = UNetSmall()
    model.load_state_dict(torch.load("model_weights/unet_small_7.pth", weights_only=True))
    model.eval()
    pathlist = []
    pathlist.append(agent_pos)
    
    while agent_pos != goal:
        input = reformat_input(rewards, obstacles_map)
        input = input.unsqueeze(0)
        V = model(input).detach().numpy().squeeze()
        policy = extract_policy(V, obstacles_map, neighbors, 10)
        next_pos = tuple(int(i) for i in policy[agent_pos])

        checker.update(agent_pos, next_pos)
        if checker.check(agent_pos, next_pos):
            print("Live Lock Detected")
            path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)
            i = 1  # Start at the first step in the heuristic path
            while i < len(path) and rewards[agent_pos] == 0:
                next_pos = path[i]
                agent_pos = next_pos
                pathlist.append(tuple(agent_pos))
                i += 1
            
            rewards[agent_pos] = 0  # Mark the current position as visited
        else:
            agent_pos = next_pos
            pathlist.append(tuple(agent_pos))
            rewards[agent_pos] = 0  # Mark the current position as visited

    return pathlist

def get_vin_path_with_value_hueristic(vin, n, obstacle_map,rewards, start,goal,k = 16):


    actions = {0:(0,-1),
               1:(1,0),
               2:(0,1),
               3:(-1,0)} # up, right, down , left

    checker = LiveLockChecker(counter=0,last_visited={})
    agent_pos = deepcopy(start)
    path = [(agent_pos,"VIN")]
    max_step = 100
    step = 0

    infrence_time = []

    while agent_pos!=goal and step<max_step:
        rewards[agent_pos[0],agent_pos[1]] = 0
        input = reformat_input(rewards,obstacle_map)
        input = input.unsqueeze(0)
        assert input.shape == (1,2,n,n) 

        start = time.time()

        logits,_,_ = vin(input.to(device),torch.tensor(agent_pos[0]).to(device),torch.tensor(agent_pos[1]).to(device),k)

        end = time.time() - start

        infrence_time.append(end)

        pred = torch.argmax(logits).item()

        action = actions[pred]

        new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])

        checker.update(agent_pos,new_pos)
        if checker.check(agent_pos,new_pos) or obstacle_map[new_pos]:
            print("Live Lock or collision detected, swithcing to A* heuristic")
            print(np.mean(infrence_time))

            heuristic_path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)

            if not heuristic_path:
                print("Heuristic failed")
                return path

            for i in range(1,len(heuristic_path)):
                next_pos = heuristic_path[i]
                agent_pos = next_pos
                path.append((agent_pos,"heuristic"))
                if agent_pos == goal:
                    break
            
            continue

            # new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])
            # checker.update(agent_pos,new_pos)

        
        agent_pos = new_pos

        path.append((agent_pos,"VIN"))

    print("mean infrence time" ,np.mean(infrence_time))
    print("total time to reach goal", np.sum(infrence_time))
    return path

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
            next_pos_policy = path[j+1][1]
            next_pos = path[j+1][0]
            cur_pos = path[j][0]
            if next_pos_policy == "heuristic":
                ax_i.annotate('', xy=(next_pos[1], next_pos[0]), xytext=(cur_pos[1], cur_pos[0]),
                        arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->'))
                
            else:
                ax_i.annotate('', xy=(next_pos[1], next_pos[0]), xytext=(cur_pos[1], cur_pos[0]),
                        arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'))
            
            # ax_i.annotate('', xy=(path[j+1][1], path[j+1][0]), xytext=(path[j][1], path[j][0]),
            #               arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'))
            
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


    # plt.savefig("images/vin_vs_vi/" + f"{seed}"+"vin_comparison_success.png", format='png')  # Save the figure as a PNG file
    plt.show()

def compute_total_reward_per_step(path, rewards):
    """
    Compute the total reward collected along a path.
    
    Args:
        path (list of tuples): List of positions (row, col) visited by the agent.
        rewards (np.ndarray): 2D array representing the initial rewards in the grid.
        
    Returns:
        total_reward (float): Total reward collected along the path.
        rewards_per_step (list): List of rewards collected at each step.
    """
    total_reward = 0
    rewards_per_step = []




    for pos,label in path:
        reward = rewards[pos[0], pos[1]]
        total_reward += reward
        rewards_per_step.append(total_reward)

        rewards[pos[0], pos[1]] = 0  # Set the reward to 0 after collection
    
    return total_reward, rewards_per_step

def compute_reward_per_step(path, rewards):
    """
    Compute the total reward collected along a path.
    
    Args:
        path (list of tuples): List of positions (row, col) visited by the agent.
        rewards (np.ndarray): 2D array representing the initial rewards in the grid.
        
    Returns:
        total_reward (float): Total reward collected along the path.
        rewards_per_step (list): List of rewards collected at each step.
    """
    total_reward = 0
    rewards_per_step = []

    for pos in path:
        reward = rewards[pos[0], pos[1]]
        rewards_per_step.append(reward)
        total_reward += reward
        rewards[pos[0], pos[1]] = 0  # Set the reward to 0 after collection
    
    return total_reward, rewards_per_step

def compute_reward_efficiency(path, rewards):
    """
    Compute the reward efficiency of a path.
    
    Args:
        path (list of tuples): List of positions (row, col) visited by the agent.
        rewards (np.ndarray): 2D array representing the initial rewards in the grid.
        
    Returns:
        reward_efficiency (float): Reward efficiency metric (total reward / number of steps).
        total_reward (float): Total reward collected along the path.
        num_steps (int): Number of steps taken along the path.
    """
    total_reward, _ = compute_reward_per_step([pos for pos, _ in path], rewards)
    num_steps = len(path)  # Number of steps is the length of the path
    reward_efficiency = total_reward / num_steps if num_steps > 0 else 0

    return reward_efficiency, total_reward, num_steps

def plot_rewards_per_step(rewards_per_step_list, titles):
    """
    Plot the rewards collected at each step for multiple paths.
    
    Args:
        rewards_per_step_list (list of lists): List of rewards collected at each step for each path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rewards_per_step in enumerate(rewards_per_step_list):
        ax.plot(rewards_per_step, label=titles[i])

    ax.set_title("Total Rewards Collected at Each Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def compute_normalized_reward_efficiency(path, rewards):
    """
    Compute the normalized reward efficiency of a path in a sparse reward setting.
    
    Args:
        path (list of tuples): List of positions (row, col) visited by the agent.
        rewards (np.ndarray): 2D array representing the initial rewards in the grid.
        
    Returns:
        normalized_efficiency (float): Normalized reward efficiency.
        total_reward (float): Total reward collected along the path.
        total_potential_reward (float): Total potential reward in the grid.
    """
    total_reward, _ = compute_reward_per_step([pos for pos, _ in path], rewards)
    total_potential_reward = np.sum(rewards[rewards > 0])

    normalized_efficiency = total_reward / total_potential_reward if total_potential_reward > 0 else 0

    return normalized_efficiency, total_reward, total_potential_reward

def compute_reward_per_encounter(path, rewards):
    """
    Compute the reward per encounter of a path.
    
    Args:
        path (list of tuples): List of positions (row, col) visited by the agent.
        rewards (np.ndarray): 2D array representing the initial rewards in the grid.
        
    Returns:
        reward_per_encounter (float): Average reward collected when reaching a reward state.
        num_encounters (int): Number of reward states encountered.
    """
    total_reward, rewards_per_step = compute_reward_per_step([pos for pos, _ in path], rewards)
    num_encounters = sum(1 for r in rewards_per_step if r > 0)

    reward_per_encounter = total_reward / num_encounters if num_encounters > 0 else 0

    return reward_per_encounter, total_reward, num_encounters


def plot_reward_metrics(total_rewards_per_path,reward_efficiency_table, normalized_efficiency_table, reward_per_encounter_table):
    """
    Plot box-and-whiskers plots for reward metrics.
    
    Args:
        reward_efficiency_table (dict): Dictionary of reward efficiency metrics per algorithm.
        normalized_efficiency_table (dict): Dictionary of normalized efficiency metrics per algorithm.
        reward_per_encounter_table (dict): Dictionary of reward per encounter metrics per algorithm.
    """
    for metric, table in zip(
        ["Total Rewards per Path","Reward Efficiency", "Normalized Efficiency", "Reward per Encounter"], 
        [total_rewards_per_path,reward_efficiency_table, normalized_efficiency_table, reward_per_encounter_table]
    ):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert the dictionary to a list of values for each algorithm
        data = [values for values in table.values()]
        labels = list(table.keys())

        # Create a box-and-whiskers plot
        ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)

        # Set plot details
        ax.set_title(f"{metric} Distribution Across Algorithms")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(f"{metric}")
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        #plt.savefig("images/puct/" + f"{metric}"+"vin_comparison.png", format='png')  # Save the figure as a PNG file


def get_reward_metrics(seeds, vin, config, device):

        T = 30
          # seeds = [9651]
        min_obstacles = 2
        max_obstacles = 10
        n = 20
        num_blocks = 5
        square_size = 10    

        reward_efficiency_table = defaultdict(list)
        nomalized_efficiency_table = defaultdict(list)
        reward_per_encounter_table = defaultdict(list)
        total_rewards_per_path = defaultdict(list)

        mean_times = []
        step_lists = []

        for seed in [14621, 12771, 15214, 13201,  8901, 13922, 10723,  8528]:
            rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, 2, 20, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
            if np.sum(rewards) == 0:
                continue

            # obstacles_map = np.zeros_like(obstacles_map)
            neighbors = precompute_next_states(n, obstacles_map)
            start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)

            if not astar_search(obstacles_map, start, goal):
                continue

            reachable = mark_reachable_cells(start,obstacles_map)
            rewards[~reachable] = 0

            vi_path, mean_inf_time, steps = get_vi_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)
            vin_path = get_vin_path_with_value(vin, n, obstacles_map, rewards.copy(), start, goal, k=50)
            vin_replan = vin_replanner(vin, n, obstacles_map, rewards.copy(), start, goal, k=50)
            vin_replan = [(p, "VIN Replan") for p in vin_replan]



            # vin_heuristic_path = get_vin_path_with_value_hueristic(vin,20,obstacles_map,rewards.copy(),start,goal,k=50)
            # puct_vin_path,mean_inf_time,total_inf_time = get_puct_path(vin,config,obstacles_map,rewards.copy(),start,goal,k=50)

            paths = [vi_path, vin_path, vin_replan]
            titles = ["Infinite Horizon VI", "VIN", "VIN Replan"]
            # titles = ["NN", "VI", "VI + NN", "Heuristic + NN","PNET","VIN"]
            # titles = ["Infinite Horizon VI" ,"VIN + Heuristic (hueristic is in white)"]

            compare_paths(paths, rewards, obstacles_map, goal, seed=seed, titles=titles)
            #plot_rewards_per_step(paths, titles)

            reward_efficiency, total_reward, num_steps = compute_reward_efficiency(vi_path, rewards.copy())
            reward_efficiency_table["VI"].append(reward_efficiency)

            reward_efficiency, total_reward, num_steps = compute_reward_efficiency(vin_path, rewards.copy())
            reward_efficiency_table["VIN"].append(reward_efficiency)

            reward_efficiency, total_reward, num_steps = compute_reward_efficiency(vin_replan, rewards.copy())
            reward_efficiency_table["VIN Replan"].append(reward_efficiency)


            # reward_efficiency, total_reward, num_steps = compute_reward_efficiency(puct_vin_path, rewards.copy())
            # reward_efficiency_table["PUCT"].append(reward_efficiency)

            normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(vi_path, rewards.copy())
            nomalized_efficiency_table["VI"].append(normalized_efficiency)

            normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(vin_path, rewards.copy())
            nomalized_efficiency_table["VIN"].append(normalized_efficiency)

            normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(vin_replan, rewards.copy())
            nomalized_efficiency_table["VIN Replan"].append(normalized_efficiency)


            # normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(puct_vin_path, rewards.copy())
            # nomalized_efficiency_table["PUCT"].append(normalized_efficiency)

            reward_per_encounter, total_reward, num_encounters = compute_reward_per_encounter(vi_path, rewards.copy())
            reward_per_encounter_table["VI"].append(reward_per_encounter)
            total_rewards_per_path["VI"].append(total_reward)

            reward_per_encounter, total_reward, num_encounters = compute_reward_per_encounter(vin_path, rewards.copy())
            reward_per_encounter_table["VIN"].append(reward_per_encounter)
            total_rewards_per_path["VIN"].append(total_reward)

            reward_per_encounter, total_reward, num_encounters = compute_reward_per_encounter(vin_replan, rewards.copy())
            reward_per_encounter_table["VIN Replan"].append(reward_per_encounter)
            total_rewards_per_path["VIN Replan"].append(total_reward)

            # reward_efficiency, total_reward, num_steps = compute_reward_per_encounter(puct_vin_path, rewards.copy())
            # reward_per_encounter_table["PUCT"].append(reward_efficiency)
            # total_rewards_per_path["PUCT"].append(total_reward)



        plot_reward_metrics(total_rewards_per_path, reward_efficiency_table, nomalized_efficiency_table, reward_per_encounter_table)
        print(np.mean(mean_times))
        print(np.mean(step_lists))        

def visualize_values_and_rewards(value_map, reward_map, obstacles, start, goal, curr_pos=None):
    """
    Visualize the value map and ground truth reward map with obstacles, start, goal, and current position.
    
    Args:
        value_map (np.ndarray): 2D array representing the value of each grid cell.
        reward_map (np.ndarray): 2D array representing the ground truth rewards of each grid cell.
        obstacles (np.ndarray): 2D binary array where 1 indicates an obstacle.
        start (tuple): Coordinates of the start position (row, col).
        goal (tuple): Coordinates of the goal position (row, col).
        curr_pos (tuple, optional): Coordinates of the current position (row, col). Default is None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Extract value map from tensor if needed
    if len(value_map.shape) == 4:  # If value map has batch/channel dimensions
        value_map = value_map[0, 0, :, :].detach().numpy()

    # Prepare the display matrix for the value map with NaN for obstacles
    display_value_map = np.copy(value_map)
    display_value_map[obstacles] = np.nan  # Set obstacles to NaN for black color

    # Plot the value map
    im1 = axes[0].imshow(display_value_map, cmap='viridis', origin='upper')
    axes[0].set_title("Value Map with Obstacles, Start, Goal, and Agent")
    fig.colorbar(im1, ax=axes[0])

    # Plot start, goal, and current positions on the value map
    axes[0].plot(start[1], start[0], 'wo', label='Start')  # Start in blue
    axes[0].plot(goal[1], goal[0], 'ro', label='Goal')    # Goal in green
    if curr_pos is not None:
        axes[0].plot(curr_pos[1], curr_pos[0], 'ro', markersize=10, label='Current')  # Current position in red

    # Prepare the display matrix for the reward map with NaN for obstacles
    display_reward_map = np.copy(reward_map)
    display_reward_map[obstacles] = np.nan  # Set obstacles to NaN for black color

    # Plot the reward map
    im2 = axes[1].imshow(display_reward_map, cmap='viridis', origin='upper')
    axes[1].set_title("Ground Truth Reward Map with Obstacles")
    fig.colorbar(im2, ax=axes[1])

    # Plot start, goal, and current positions on the reward map
    axes[1].plot(start[1], start[0], 'wo', label='Start')  # Start in blue
    axes[1].plot(goal[1], goal[0], 'ro', label='Goal')    # Goal in green
    if curr_pos is not None:
        axes[1].plot(curr_pos[1], curr_pos[0], 'ro', markersize=10, label='Current')  # Current position in red

    # Adjust plot settings
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def test_value_cnn(config):

    n = config.n
    rewards, obstacles_map = init_random_reachable_map(n, config.obstacle_shape, config.num_obstacles, config.min_obstacles, config.max_obstacles, config.obstacle_type, config.square_size, config.obstacle_map, config.seed, config.num_reward_blocks, config.reward_square_size, config.obstacle_cluster_prob, config.obstacle_square_sizes)
    neighbors = precompute_next_states(n, obstacles_map)
    start, goal = pick_start_and_goal(rewards, obstacles_map, seed=config.seed)

    if not astar_search(obstacles_map, start, goal):
        print("No path found")
        return

    env = ModifiedGridEnvironment(config, rewards, obstacles_map, start, goal)

    model = UNetSmall()

    model.load_state_dict(torch.load("/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/model_weights/smallfinal_model.pt", weights_only=True, map_location=torch.device('mps')))

    env.reset()

    input, state_x, state_y = env.get_vin_input()

    V = model(input)

    visualize_values_and_rewards(V, rewards, obstacles_map, start, goal)


def vin_replanner(vin, n, obstacle_map,rewards, start,goal,k = 50):

    empty_map = np.zeros_like(obstacle_map)

    vin_path = get_vin_path_with_value(vin, n, empty_map, rewards, start, goal, k)
    vin_path = [pos for pos, _ in vin_path]

    current = start
    i = 0
    path = []
    # Get positions of VIN path that correspond to obstacles
    while i < len(vin_path):
        next_pos = vin_path[i]

        if obstacle_map[next_pos] == 1:
            # Find the next non-obstacle position in the VIN path
            j = i + 1
            while j < len(vin_path) and obstacle_map[vin_path[j]] == 1:
                j += 1

            if j == len(vin_path):
                # No valid position found; end the path
                break

            next_non_obstacle = vin_path[j]
            # Use A* to navigate to the next non-obstacle position
            a_star_path = astar_search(obstacle_map,current, next_non_obstacle)
            if not a_star_path:
                # If A* fails, break
                break

            # Add A* path to the final path (exclude the last position to avoid duplication)
            path.extend(a_star_path[:-1])
            current = next_non_obstacle
            i = j  # Skip to the next non-obstacle position in the VIN path
        else:
            # No obstacle; add the position directly
            path.append(next_pos)
            current = next_pos
            i += 1

    return path




def mark_reachable_cells(start, obstacles_map):
    n = obstacles_map.shape[0]
    reachable = np.zeros_like(obstacles_map, dtype=bool)
    queue = deque([start])
    reachable[start] = True

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Down, Right, Up

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not obstacles_map[nx, ny] and not reachable[nx, ny]:
                reachable[nx, ny] = True
                queue.append((nx, ny))

    return reachable


def visualize_replanner(vin,seed):
    n = 20
    min_obstacles = 2
    max_obstacles = 10
    n = 20
    num_blocks = 5
    square_size = 10  
    rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, 2, 20, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
    start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)

    # reachable = mark_reachable_cells(start, obstacles_map)
    # rewards[~reachable] = 0  # Set rewards to zero for unreachable cells

    vin_path = vin_replanner(vin, n, obstacles_map, rewards.copy(), start, goal, k=50)



    for p in vin_path:
        rewards[p] = 0
        visualize_rewards(rewards,obstacles_map,p,goal)
        if p == goal:
            break




    


    

  
        

            







if __name__ == "__main__":
    from types import SimpleNamespace
    from astar import *
    from collections import defaultdict
    import argparse


    # seeds = np.random.randint(6100,20000,10)
    seeds = [16515]
    # seeds = [1234]

    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True,map_location=device)
    #vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/model_weights/vin_full_traj.pth')
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
    parser.add_argument('--seed', type=int, default=5324234, help='Random seed')
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
    vin.eval()
    vin.to(device)

    # get_reward_metrics(seeds, vin, config, device)

    for seed in seeds:
        visualize_replanner(vin,seed)













    

