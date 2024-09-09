import numpy as np
import random 
import torch
from utils import *
import pickle
from fo_solver import visualize_rewards,visualize_policy_and_rewards
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from astar import astar_search

def visualize_center_of_mass(rewards, obstacles, start, goal, curr_pos=None, next_pos=None, centers_of_mass=None,path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the reward function
    display_matrix = np.copy(rewards)
    display_matrix[obstacles] = np.nan  # Set obstacles to NaN for black color
    im = ax.imshow(display_matrix, cmap='viridis', origin='upper')
    ax.set_title("Policy with Rewards and Obstacles in Background")
    fig.colorbar(im, ax=ax)

    ax.plot(start[1], start[0], 'bo')
    ax.plot(goal[1], goal[0], 'go')

    if curr_pos is not None:
        ax.plot(curr_pos[1], curr_pos[0], 'ro', markersize=10)
        i,j = curr_pos



    if curr_pos is not None:
        if next_pos[0] < curr_pos[0]:
            ax.arrow(j, i, 0, -0.5, head_width=0.2, head_length=0.2, fc='r', ec='r')
        if next_pos[0] > curr_pos[0]:
            ax.arrow(j, i, 0, 0.5, head_width=0.2, head_length=0.2, fc='r', ec='r')
        if next_pos[1] < curr_pos[1]:
            ax.arrow(j, i, -0.5, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
        if next_pos[1] > curr_pos[1]:
            ax.arrow(j, i, 0.5, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')

    if centers_of_mass is not None:
        for coord in centers_of_mass:
            ax.plot(coord[1], coord[0], 'rx', markersize=10)

    if path is not None:
        for i in range(len(path) - 1):
            curr_pos = path[i]
            next_pos = path[i + 1]
            x1, y1 = curr_pos[1], curr_pos[0]
            x2, y2 = next_pos[1], next_pos[0]
            dx, dy = x2 - x1, y2 - y1
            ax.arrow(x1, y1, dx, dy, head_width=0.2, head_length=0.2, fc='r', ec='r',width=0.05)

    ax.invert_yaxis()
    ax.grid(True)
    plt.show()


def center_of_mass_heuristc(obstacle_map,rewards, agent_pos):
    """
    If the agent is in a like lock situation, take a couple of steps towards the center of mass of the reward cluster according to an a star search with l1 heuristic.
    """
    

    reward_label_array, num_features = label(rewards>0)
    centers_of_mass = center_of_mass(rewards, reward_label_array, range(1,num_features+1))
    CoM = {i:np.round(centers_of_mass[i-1]).astype(int) for i in range(1,num_features+1)}
    obstacle_label_array, num_obstacles = label(obstacle_map)

    mass_map = {} # cluster label : mass

    for i in range(1,num_features+1):
        mass_map[i] = np.sum(rewards[reward_label_array == i])

    # find the cluster with the highest mass
    
    gravitational_pull = {}
    for i in range(1,num_features+1):
        gravitational_pull[i] = mass_map[i]/np.linalg.norm(agent_pos - CoM[i],2)
    
    path = astar_search(obstacle_map,agent_pos,tuple(CoM[max(gravitational_pull,key=gravitational_pull.get)]))

    return path


def linear_interpolation(agent_pos,neighbors,V):
    neighbors = neighbors[agent_pos] # list of neighbors
    V_neighbors = [V[neighbor] for neighbor in neighbors]

    return np.mean(V_neighbors)


def k_nearest_neighbors():
    pass

def random_walk(agent_pos,neighbors):
    return random.choice(neighbors[agent_pos])


def calculate_distances(obstacles_map, centers_of_mass):
    rows, cols = obstacles_map.shape
    x_indices, y_indices = np.indices((rows, cols))
    print(CoM)
    # Initialize the distance array
    distances = np.zeros((len(centers_of_mass.keys()), rows, cols))
    
    for idx, (x_center, y_center) in centers_of_mass.items():
        # if obstacles_map[x_center, y_center]:
        #     continue
        distances[idx] = np.sqrt((y_indices - y_center)**2 + (x_indices - x_center)**2)
    
    return distances


def calculate_potential_field(obstacle_map, distances, mass_map, influence_factor=1):
    potential_field = np.zeros(obstacle_map.shape)
    
    for idx in range(distances.shape[0]):
        potential_field += (influence_factor * mass_map[idx]) / (distances[idx] + 1e-5)  # Adding small value to avoid division by zero
    
    return potential_field


    





if __name__ == "__main__":
    import copy
    from dl_models import UNetSmall
    from eval import LiveLockChecker,reformat_input
    from fo_solver import extract_policy,value_iteration



    with open("obstacle.pkl","rb") as f:
        obstacle_map = pickle.load(f)


    rewards, obstacle_map = init_map(n=10, config="block", num_blocks=3, num_obstacles=3,square_size=10,obstacle_map=obstacle_map)


    # reward_label_array, num_features = label(rewards>0)
    # centers_of_mass = center_of_mass(rewards, reward_label_array, range(1,num_features+1))
    # CoM = {i:np.round(centers_of_mass[i-1]).astype(int) for i in range(1,num_features+1)}

    # mass_map = {} # cluster label : mass

    # for i in range(1,num_features+1):
    #     mass_map[i] = np.sum(rewards[reward_label_array == i])

    # distances = calculate_distances(obstacle_map, CoM)

    neighbors = precompute_next_states(n, obstacle_map)
    start, goal = pick_start_and_goal(rewards, obstacle_map)
    init_rewards = np.copy(rewards) 



    agent_pos = copy.deepcopy(start)
    checker = LiveLockChecker(counter=0,last_visited={})
    model = UNetSmall()
    model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))
    model.eval()
    while agent_pos != goal:
        input = reformat_input(rewards, obstacle_map)
        input = input.unsqueeze(0)
        V= model(input).detach().numpy().squeeze()
        policy = extract_policy(V,obstacle_map,neighbors,10)
        next_pos = tuple(int(i) for i in policy[agent_pos])

        checker.update(agent_pos,next_pos)
        if checker.check(agent_pos,next_pos):
            print("Live Lock Detected")

            v_local = linear_interpolation(agent_pos,neighbors,V)
            print(f"Current Value: {V[agent_pos]}")
            print(f"Local Value: {v_local}")
            prev_pos = copy.deepcopy(agent_pos)
            visualize_rewards(rewards, obstacle_map, start, goal, agent_pos, next_pos)
            agent_pos = next_pos
            rewards[agent_pos] = 0

            # V  = model(reformat_input(rewards,obstacle_map).unsqueeze(0)).detach().numpy().squeeze()
f
            V[prev_pos] = v_local
            policy = extract_policy(V,obstacle_map,neighbors,10)
            next_pos = tuple(int(i) for i in policy[agent_pos])
            visualize_rewards(rewards, obstacle_map, start, goal, agent_pos, next_pos)
            agent_pos = next_pos
            rewards[agent_pos] = 0
            
        else:
            visualize_rewards(rewards, obstacle_map, start, goal, agent_pos, next_pos)
            agent_pos = next_pos
            rewards[agent_pos] = 0
        
 

        
        











