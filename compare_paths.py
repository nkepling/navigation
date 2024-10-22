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

def get_puct_path(vin, n, env, start, goal, k=50, c_puct=1.0):
    pass


def density_aware_vi(n, rewards, obstacles_map, neighbors, start, goal):
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 10000
    step = 0

    while agent_pos != goal and step < max_step:
        rewards[agent_pos] = 0
        density_reward_map = create_density_based_reward_map(rewards,0.2,1)
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

        logits,_ = vin(input,torch.tensor(agent_pos[0]),torch.tensor(agent_pos[1]),k)

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

    print("mean infrence time" ,np.mean(infrence_time))
    print("total time to reach goal", np.sum(infrence_time))
    return path


def get_vin_path_with_value(vin, n, obstacle_map,rewards, start,goal,k = 16):


    actions = {0:(0,-1),
               1:(1,0),
               2:(0,1),
               3:(-1,0)} # up, right, down , left

    checker = LiveLockChecker(counter=0,last_visited={})
    agent_pos = deepcopy(start)
    path = [agent_pos]
    max_step = 100
    step = 0

    infrence_time = []

    while agent_pos!=goal and step<max_step:
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
        if checker.check(agent_pos,new_pos):
            print("Live Lock Detected")
            print(np.mean(infrence_time))
            break
            # path = center_of_mass_heuristic(obstacles_map, rewards, agent_pos)


            # new_pos = tuple([agent_pos[0] + action[0],agent_pos[1]+action[1]])
            # checker.update(agent_pos,new_pos)

        
        agent_pos = new_pos

        path.append(agent_pos)

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

        logits,_,_ = vin(input,torch.tensor(agent_pos[0]),torch.tensor(agent_pos[1]),k)

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

        # plt.savefig("images/vin_vs_vi/" + f"{metric}"+"vin_comparison.png", format='png')  # Save the figure as a PNG file







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





if __name__ == "__main__":
    from types import SimpleNamespace
    from astar import *
    from collections import defaultdict


    seeds = np.random.randint(6100,20000,3)

    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True,map_location=device)

    config = SimpleNamespace(k = 25, 
                            l_i = 2,
                            l_q = 4,
                            l_h = 150,
                            imsize = 10,
                            batch_sz = 1
                            )

    vin = VIN(config)
    vin.load_state_dict(vin_weights)
    vin.eval()

    T = 30

    min_obstacles = 2
    max_obstacles = 20
    n = 20

    reward_efficiency_table = defaultdict(list)
    nomalized_efficiency_table = defaultdict(list)  
    reward_per_encounter_table = defaultdict(list)  
    total_rewards_per_path = defaultdict(list)

    for seed in seeds:
        # square_size = random.randint(2,8)
        # num_blocks = random.randint(3,6)
        # rewards, obstacles_map = init_map(n, "block", num_blocks, num_obstacles, obstacle_type, square_size,seed=seed)
        rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, min_obstacles, max_obstacles, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
        if np.sum(rewards) == 0:
            continue
        #rewards, obstacles_map = init_reachable_map(n, "block", num_blocks, num_obstacles, obstacle_type, seed=seed)
        neighbors = precompute_next_states(n, obstacles_map)
        start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)

        if not astar_search(obstacles_map,start,goal):
            continue


        visualize_rewards(rewards,obstacles_map,start,goal)

        visualize_rewards(create_density_based_reward_map(rewards),obstacles_map,start,goal)

    

        #model = UNetSmall()
        #model.load_state_dict(torch.load("model_weights/unet_small_7.pth",weights_only=True))
        #input = reformat_input(rewards, obstacles_map)
        #input = input.unsqueeze(0)


        #V= model(input).detach().numpy().squeeze()

        #Viter = value_iteration(n, rewards.copy(), obstacles_map, gamma,neighbors)
        #V_init = nn_initialized_vi(model,n, rewards.copy(), obstacles_map, gamma,neighbors)

        #path1 = get_nn_path(n, rewards.copy(), obstacles_map, nqeighbors, start, goal, model)
        path2 = get_vi_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)

        vi_total_reward,vi_total_reward_per_step = compute_total_reward_per_step(path2,rewards.copy())

        total_rewards_per_path["VI"].append(vi_total_reward)
        #path3 = get_vi_plus_nn_path(n, rewards.copy(), obstacles_map, neighbors, start, goal,model)q
        #path4 = get_heuristic_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)
        #path5 = get_p_net_path(cae_net,cae_net_path,pnet,pnet_path,obstacles_map,rewards.copy(),start,goal)
        path6 = get_vin_path_with_value_hueristic(vin,20,obstacles_map,rewards.copy(),start,goal,k=50)

        heuristic_total_reward,heuristic_total_reward_per_step = compute_total_reward_per_step(path6,rewards.copy())
        total_rewards_per_path["VIN"].append(heuristic_total_reward)



        path7 = density_aware_vi(n,rewards.copy(),obstacles_map,neighbors,start,goal)

        total_rewards_per_path["Density Aware VI"].append(compute_total_reward_per_step(path7,rewards.copy())[0])



        
        # path7 = get_finite_vi_path(n,rewards.copy(),obstacles_map,neighbors,T,start,goal)
        # input = reformat_input(rewards,obstacles_map)
        # input = input.unsqueeze(0)
        # logits,pred,values = vin(input,torch.tensor(0),torch.tensor(0),50)
        # visualize_values_and_rewards(values,rewards,obstacles_map,start,goal)

        

        

        paths = [path2,path6,path7]
        # titles = ["NN", "VI", "VI + NN", "Heuristic + NN","PNET","VIN"]
        titles = ["Infinite Horizon VI" ,"VIN","Density Aware VI"]

        compare_paths(paths, rewards, obstacles_map, goal,seed=seed,titles=titles)
        #plot_rewards_per_step([vi_total_reward_per_step,heuristic_total_reward_per_step],titles)

        # reward_efficiency, total_reward, num_steps = compute_reward_efficiency(path2,rewards.copy())
        # reward_efficiency_table["VI"].append(reward_efficiency)

        # reward_efficiency, total_reward, num_steps = compute_reward_efficiency(path6,rewards.copy())
        # reward_efficiency_table["VIN"].append(reward_efficiency)

        # reward_efficiency, total_reward, num_steps = compute_reward_efficiency(path7,rewards.copy())
        # reward_efficiency_table["Density Aware VI"].append(reward_efficiency)

        # normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(path2,rewards.copy())
        # nomalized_efficiency_table["VI"].append(normalized_efficiency)

        # normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(path6,rewards.copy())
        # nomalized_efficiency_table["VIN"].append(normalized_efficiency)

        # normalized_efficiency, total_reward, total_potential_reward = compute_normalized_reward_efficiency(path7,rewards.copy())
        # nomalized_efficiency_table["Density Aware VI"].append(normalized_efficiency)

        
        # reward_per_encounter, total_reward, num_encounters = compute_reward_per_encounter(path2,rewards.copy())
        # reward_per_encounter_table["VI"].append(reward_per_encounter)

        # reward_per_encounter, total_reward, num_encounters = compute_reward_per_encounter(path6,rewards.copy())
        # reward_per_encounter_table["VIN"].append(reward_per_encounter)

        # reward_efficiency, total_reward, num_steps = compute_reward_per_encounter(path7,rewards.copy())
        # reward_per_encounter_table["Density Aware VI"].append(reward_efficiency)   

    # plot_reward_metrics(total_rewards_per_path,reward_efficiency_table, nomalized_efficiency_table, reward_per_encounter_table)






    

    
