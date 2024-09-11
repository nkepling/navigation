import numpy as np
import matplotlib.pyplot as plt
from utils import *
import heuristics
from heuristics import *
from nn_training import *
from fo_solver import *
import pickle
import torch
from eval import *
from dl_models import *


# def get_heuristic_path(n, rewards, obstacles_map, neighbors, start, goal):
#     agent_pos = deepcopy(start)
#     checker = LiveLockChecker(counter=0,last_visited={})
#     model = UNetSmall()
#     model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))
#     model.eval()
#     pathlist = []
#     pathlist.append(agent_pos)
#     while agent_pos != goal:
#         input = reformat_input(rewards, obstacles_map)
#         input = input.unsqueeze(0)
#         V= model(input).detach().numpy().squeeze()
#         policy = extract_policy(V,obstacles_map,neighbors,10)
#         next_pos = tuple(int(i) for i in policy[agent_pos])

#         checker.update(agent_pos,next_pos)
#         if checker.check(agent_pos,next_pos):
#             print("Live Lock Detected")
#             path = center_of_mass_heuristic(obstacles_map,rewards,agent_pos)
#             i = 1
#             while rewards[agent_pos] == 0:
#                 next_pos = path[i]
#                 #visualize_rewards(rewards,obstacles_map,start,goal,agent_pos,next_pos)
#                 agent_pos = next_pos
#                 pathlist.append(tuple(agent_pos))
#                 i+=1
                
#             # next_pos = path[1]
#             # visualize_rewards(rewards,obstacle_map,start,goal,agent_pos,next_pos)
#             # agent_pos = next_pos

#             rewards[agent_pos] = 0
#         else:
#             #visualize_rewards(rewards, obstacles_map, start, goal, agent_pos, next_pos)
#             agent_pos = next_pos
#             pathlist.append(tuple(agent_pos))
#             rewards[agent_pos] = 0
#         # path.append(agent_pos)
#     return path

def get_heuristic_path(n, rewards, obstacles_map, neighbors, start, goal):
    agent_pos = deepcopy(start)
    checker = LiveLockChecker(counter=0, last_visited={})
    model = UNetSmall()
    model.load_state_dict(torch.load("model_weights/unet_small_5.pth", weights_only=True))
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


    plt.savefig(f"{seed}"+"path_comparison_final.png", format='png')  # Save the figure as a PNG file
    plt.show()





if __name__ == "__main__":
    seeds = np.random.randint(0,1000,10)
    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)
    for seed in seeds:
        rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)
        neighbors = precompute_next_states(n, obstacles_map)
        start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)
        model = UNetSmall()
        model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))
        input = reformat_input(rewards, obstacles_map)
        input = input.unsqueeze(0)
        print(input.shape)

        V= model(input).detach().numpy().squeeze()

        Viter = value_iteration(n, rewards.copy(), obstacles_map, gamma,neighbors)
        V_init = nn_initialized_vi(model,n, rewards.copy(), obstacles_map, gamma,neighbors)

        path1 = get_nn_path(n, rewards.copy(), obstacles_map, neighbors, start, goal, model)
        path2 = get_vi_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)
        path3 = get_vi_plus_nn_path(n, rewards.copy(), obstacles_map, neighbors, start, goal,model)
        path4 = get_heuristic_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)
        print(path4)
        paths = [path1,path2,path3,path4]
        titles = ["NN", "VI", "VI + NN", "Heuristic + NN"]

        compare_paths(paths, rewards, obstacles_map, goal,seed=seed,titles=titles)