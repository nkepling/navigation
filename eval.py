from utils import *
from fo_solver import value_iteration, extract_policy, visualize_policy_and_rewards, visualize_rewards, pick_start_and_goal
from copy import deepcopy
from nn_training import *
import matplotlib.pyplot as plt
import pickle
import torch
from dataclasses import dataclass
import time 
from io import BytesIO
from csv import writer 
from vi_with_nn_init import nn_initialized_vi
from collections import defaultdict
from dl_models import UNetSmall


def get_nn_path(n, rewards, obstacles_map, neighbors, start, goal, model):
    agent_position = deepcopy(start)
    steps = 0
    path = [agent_position]
    checker = LiveLockChecker(last_visited={}, counter=0)
    while agent_position!=goal:
        rewards[agent_position[0], agent_position[1]] = 0
        Vnn = model(reformat_input(rewards, obstacles_map).unsqueeze(0))
        Vnn = Vnn.detach().numpy().squeeze()
        policy = extract_policy(Vnn, obstacles_map,neighbors)
        next_position = tuple(int(i) for i in policy[agent_position])
        checker.update(agent_position, next_position)
        if checker.check(agent_position, next_position):
            break
        agent_position = next_position
        path.append(agent_position)
        steps += 1
    return path

def get_vi_path(n, rewards, obstacles_map, neighbors, start, goal):
    agent_position = deepcopy(start)
    steps = 0
    path = [agent_position]
    reward_list = []
    checker = LiveLockChecker(last_visited={}, counter=0)
    while agent_position!=goal:
        rewards[agent_position[0], agent_position[1]] = 0

        Viter = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
        policy = extract_policy(Viter, obstacles_map,neighbors)
        next_position = tuple(int(i) for i in policy[agent_position])
        checker.update(agent_position, next_position)
        if checker.check(agent_position, next_position):
            break
        agent_position = next_position
        path.append(agent_position)
        reward_list.append(rewards)
        steps += 1
    return path


def get_vi_plus_nn_path(n, rewards, obstacles_map, neighbors, start, goal,model):
    agent_position = deepcopy(start)
    steps = 0
    path = [agent_position]
    checker = LiveLockChecker(last_visited={}, counter=0)
    while agent_position!=goal:
        rewards[agent_position[0], agent_position[1]] = 0
        Viter = nn_initialized_vi(model,n, rewards, obstacles_map, gamma,neighbors)
        policy = extract_policy(Viter, obstacles_map,neighbors)
        next_position = tuple(int(i) for i in policy[agent_position])
        checker.update(agent_position, next_position)
        if checker.check(agent_position, next_position):
            break
        agent_position = next_position
        path.append(agent_position)
        steps += 1
    return path

# def get_heuristic_path(n, rewards, obstacles_map, neighbors, start, goal):
#     agent_pos = deepcopy(start)
#     checker = LiveLockChecker(counter=0,last_visited={})
#     model = UNetSmall()
#     model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))
#     model.eval()
#     path = [agent_pos]
#     while agent_pos != goal:
#         input = reformat_input(rewards, obstacle_maps)
#         input = input.unsqueeze(0)
#         V= model(input).detach().numpy().squeeze()
#         policy = extract_policy(V,obstacle_map,neighbors,10)
#         next_pos = tuple(int(i) for i in policy[agent_pos])

#         checker.update(agent_pos,next_pos)
#         if checker.check(agent_pos,next_pos):
#             print("Live Lock Detected")
#             path = center_of_mass_heuristc(obstacle_map,rewards,agent_pos)
#             i = 1
#             while rewards[agent_pos] == 0:
#                 next_pos = path[i]
#                 visualize_rewards(rewards,obstacle_map,start,goal,agent_pos,next_pos)
#                 agent_pos = next_pos
#                 path.append(agent_pos)
#                 i+=1
                
#             # next_pos = path[1]
#             # visualize_rewards(rewards,obstacle_map,start,goal,agent_pos,next_pos)
#             # agent_pos = next_pos

#             rewards[agent_pos] = 0
#         else:
#             visualize_rewards(rewards, obstacle_map, start, goal, agent_pos, next_pos)
#             agent_pos = next_pos
#             path.append(agent_pos)
#             rewards[agent_pos] = 0
#     return path

def visually_compare_value_functions(value_functions, paths, rewards, obstacles_map, target_location, titles=None,seed=None):
    num_vf = len(value_functions)
    fig, ax = plt.subplots(2, num_vf, figsize=(14 + 5 * (num_vf - 2), 10))  # Adjust figure size based on number of value functions

    vmin = min([np.min(V) for V in value_functions])
    vmax = max([np.max(V) for V in value_functions])

    if titles is None:
        titles = [f"Value Function {i + 1}" for i in range(num_vf)]

    for i in range(num_vf):
        ax_i = ax[0, i] if num_vf > 1 else ax[0]  # Handle single plot case
        display_matrix = np.copy(value_functions[i])

        im = ax_i.imshow(display_matrix, vmin=vmin, vmax=vmax, cmap='viridis', origin='upper')
        ax_i.plot(target_location[1], target_location[0], 'ro')
        ax_i.plot(0, 0, 'wo')

        path = paths[i]
        for j in range(len(path) - 1):
            ax_i.annotate('', xy=(path[j+1][1], path[j+1][0]), xytext=(path[j][1], path[j][0]),
                          arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'))
        ax_i.set_title(titles[i])
        # fig.colorbar(im, ax=ax_i)

        ax_i.invert_yaxis()
        ax_i.grid(True)

    fig.colorbar(im, ax=ax[0, :], orientation='vertical', fraction=0.025, pad=0.04)

    # Plot the rewards on the second row
    ax_r = ax[1, int(num_vf / 2)]  # Center the rewards plot

    display_matrix = np.copy(rewards)
    display_matrix[obstacles_map] = np.nan  # Set obstacles to NaN for black color
    im_r = ax_r.imshow(display_matrix, cmap='viridis', origin='upper')
    ax_r.plot(target_location[1], target_location[0], 'ro')
    ax_r.plot(0, 0, 'wo')
    ax_r.set_title("Ground Truth Rewards")
    fig.colorbar(im_r, ax=ax_r, orientation='vertical', fraction=0.025, pad=0.04)

    ax_r.invert_yaxis()
    ax_r.grid(True)

    # Hide any unused subplots in the second row
    for j in range(num_vf):
        if j != int(num_vf / 2):
            ax[1, j].axis('off')

    # ax_r.invert_yaxis()
    # ax_r.grid(True)
    if seed is not None:
        plt.suptitle(f"Comparison of Value Functions and Paths (Seed: {seed})")
    
    # plt.savefig(f"value_function_comparison_seed_{seed}.png")
    plt.show()

def compare_paths(paths, rewards, obstacles_map, target_location,seed=None):
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
        ax_i.set_title(f"Path {i + 1}")
        # fig.colorbar(im, ax=ax_i)

        ax_i.invert_yaxis()
        ax_i.grid(True)

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.04)
    if seed is not None:
        plt.suptitle(f"Comparison of Paths (Seed: {seed})")
    plt.show()



def visually_compare_policy(V1,V2,rewards,obstacle_map,neighbors,v1name="Value Function 1",v2name="Value Function 2"):
    fig, ax = plt.subplots(1, 3,figsize=(14, 10)) 
    ax1,ax2,ax3 = ax
    display_matrix = np.copy(V1)

    dirs = {0:"up",1:"down",2:"left",3:"right"}
    dir_map = {
        (0, -1): 0,
        (0, 1): 1,
        (-1, 0): 2,
        (1, 0): 3
    }

    p1 = extract_policy(V1, rewards, neighbors)
    p2 = extract_policy(V2, rewards, neighbors)

    p1_dir = np.zeros_like(p1)
    p2_dir = np.zeros_like(p2)

    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            coord = np.array([i,j])
            dir1 = tuple(p1[i,j] - coord)
            dir2 = tuple(p2[i,j] - coord)
            p1_dir[i,j] = dir_map[dir1]
            p2_dir[i,j] = dir_map[dir2]

    display_matrix = np.copy(p1_dir)
    display_matrix[obstacle_map] = np.nan  # Set obstacles to NaN for black color
    im1 = ax1.imshow(display_matrix, cmap='viridis', origin='upper')

    ax1.set_title(v1name)
    fig.colorbar(im1, ax=ax1)
    display_matrix = np.copy(p2_dir)
    display_matrix[obstacle_map] = np.nan  # Set obstacles to NaN for black color
    im2 = ax2.imshow(display_matrix, cmap='viridis', origin='upper')
    ax2.set_title(v2name)
    fig.colorbar(im2, ax=ax2)

    ax1.invert_yaxis()
    ax1.grid(True)
    ax2.invert_yaxis()
    ax2.grid(True)
    ax3.invert_yaxis()
    ax3.grid(True)
    plt.show()

@dataclass
class LiveLockChecker:
    """Check if the agent is in a live lock
    """
    last_visited: dict
    counter: int = 0

    def check(self, agent_position, next_position):
        if next_position in self.last_visited:
            if self.last_visited[next_position] == agent_position and self.last_visited[agent_position] == next_position:
                self.counter += 1
                # print(f"Live Lock Detected {self.counter}")
                if self.counter > 5:
                    # print("Breaking Live Lock")
                    return True
            else:
                self.counter = 0
        return False
    
    def update(self, agent_position, next_position):
        self.last_visited[agent_position] = next_position
    


def get_comparison_metrics(seeds):
    """Get comparison metrics between Value Iteration and Neural network
    """
    # model = DeeperValueIterationModel()
    # model.load_state_dict(torch.load("model_weights/value_function_fixed_map_2.pth",weights_only=True))
    # model.eval()
    model = UNetSmall()
    model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))
    model.eval()

    file = open("unet_compare.csv", "w")    
    csv_writer = writer(file)



    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)

    for ind,seed in enumerate(seeds):
        row = {}
        ######### Value Iteration #########
        row["function"] = "Value Iteration"

        rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)
        init_rewards = deepcopy(rewards)

        neighbors = precompute_next_states(n, obstacles_map)

        start, goal = pick_start_and_goal(rewards, obstacles_map)

        checker = LiveLockChecker(last_visited={}, counter=0)

        agent_position = deepcopy(start)
        steps = 0
        row["live_lock"] = False
        row["seed"] = seed
        infrence_time =[]
        while agent_position!=goal:
            rewards[agent_position[0], agent_position[1]] = 0
            start_time = time.time()
            Viter = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
            end_time = time.time()
            infrence_time.append(end_time - start_time)
            policy = extract_policy(Viter, obstacles_map,neighbors)
            next_position = tuple(int(i) for i in policy[agent_position])

            checker.update(agent_position, next_position)
            if checker.check(agent_position, next_position):
                row["live_lock"] = True
                break
        
            agent_position = next_position
            steps += 1

        row["infrence_time"] = np.mean(infrence_time)
        row["found_goal"] = agent_position == goal
        row["steps_to_goal"] = steps

        # metrics_table = metrics_table._append(row,ignore_index=True)
        csv_writer.writerow(row.values())

        #### Neural Network + value iter ####
        row["function"] = "Neural Network + Value Iteration"
    
        # rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)
        rewards = deepcopy(init_rewards)

        neighbors = precompute_next_states(n, obstacles_map)

        checker = LiveLockChecker(last_visited={}, counter=0)

        agent_position = deepcopy(start)
        steps = 0
        row["live_lock"] = False
        row["seed"] = seed
        infrence_time =[]
        while agent_position!=goal:
            rewards[agent_position[0], agent_position[1]] = 0
            start_time = time.time()
            # Vnn = model(reformat_input(rewards, obstacles_map))
            Vnn = nn_initialized_vi(model,n, rewards, obstacles_map, gamma,neighbors)
            end_time = time.time()
            infrence_time.append(end_time - start_time)
            # Vnn = Vnn.detach().numpy().squeeze()
            policy = extract_policy(Vnn, obstacles_map,neighbors)
            next_position = tuple(int(i) for i in policy[agent_position])
            checker.update(agent_position, next_position)
            if checker.check(agent_position, next_position):
                row["live_lock"] = True
                break
        
            agent_position = next_position
            steps += 1

        row["infrence_time"] = np.mean(infrence_time)   
        row["found_goal"] = agent_position == goal
        row["steps_to_goal"] = steps
        csv_writer.writerow(row.values())
        
        #### Neural Network ####
        row["function"] = "Neural Network"
    
        # rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)
        rewards = deepcopy(init_rewards)

        neighbors = precompute_next_states(n, obstacles_map)

        checker = LiveLockChecker(last_visited={}, counter=0)

        agent_position = deepcopy(start)
        steps = 0
        row["live_lock"] = False
        row["seed"] = seed
        infrence_time =[]
        while agent_position!=goal:
            rewards[agent_position[0], agent_position[1]] = 0
            start_time = time.time()
            Vnn = model(reformat_input(rewards, obstacles_map).unsqueeze(0))
            # Vnn = nn_initialized_vi(model,n, rewards, obstacles_map, gamma,neighbors)
            end_time = time.time()
            infrence_time.append(end_time - start_time)
            Vnn = Vnn.detach().numpy().squeeze()
            policy = extract_policy(Vnn, obstacles_map,neighbors)
            next_position = tuple(int(i) for i in policy[agent_position])
            checker.update(agent_position, next_position)
            if checker.check(agent_position, next_position):
                row["live_lock"] = True
                break
        
            agent_position = next_position
            steps += 1

        row["infrence_time"] = np.mean(infrence_time)   
        row["found_goal"] = agent_position == goal
        row["steps_to_goal"] = steps

        csv_writer.writerow(row.values())
        if ind % 100 == 0:
            print(f"Seed {ind} completed")

    file.close()



def plot_infrence_time_versus_state_size():
    seeds = np.random.randint(0,1000,3)
    grid_size = [10,50,100,300,500,700,1000]

    grid_times = defaultdict(list)
    for s in tqdm(grid_size, desc="Grid Size", unit="size"):
        for seed in tqdm(seeds, desc="Seeds", unit="seed", leave=False):
            rewards, obstacles_map = init_map(s, config, num_blocks, num_obstacles, obstacle_type, square_size,seed=seed)
            neighbors = precompute_next_states(s, obstacles_map)
            start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)
            
            time_start = time.time()
            Viter = value_iteration(s, rewards, obstacles_map, gamma,neighbors)
            time_end = time.time()

            grid_times[s].append(time_end - time_start)

    grid_times = {k:np.mean(v) for k,v in grid_times.items()}
    std_errs = {k:np.std(v)/np.sqrt(len(seeds)) for k,v in grid_times.items()}
    plt.errorbar(grid_times.keys(), grid_times.values(), yerr=list(std_errs.values()), fmt='-o', capsize=5, capthick=2, elinewidth=1)
    plt.xlabel("Grid Size")
    plt.ylabel("Time (s)")

    plt.show()


def plot_change_in_value():

    s = 40
    V_prev = np.zeros((s,s))
    seed = np.random.randint(0,1000)
    # with open('obstacle.pkl', 'rb') as f:
    #     obstacles_map = pickle.load(f)

    rewards, obstacles_map = init_map(s, config, num_blocks, num_obstacles, obstacle_type, square_size,seed=seed)
    neighbors = precompute_next_states(s, obstacles_map)
    start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)
    agent_position = deepcopy(start)
    while agent_position!=goal:
        # mark current position as 0 reward
        rewards[agent_position[0], agent_position[1]] = 0
        V = value_iteration(s, rewards, obstacles_map, gamma,neighbors)
        # V = nn_initialized_vi(model,n, rewards, obstacles_map, gamma,neighbors)
        delta_V = V - V_prev
        # visualize_rewards(V, obstacles_map, agent_position, goal)
        visualize_rewards(delta_V, obstacles_map, agent_position, goal)
        V_prev = V 
        policy = extract_policy(V, obstacles_map,neighbors,n=s)
        next_position = tuple(int(i) for i in policy[agent_position])
        print("Agent next state is {}".format(next_position))
        i, j = agent_position[0], agent_position[1]
        # visualize_rewards(rewards, obstacles_map, start, goal, agent_qposition, next_position)
        agent_position = next_position


#TODO add delta for V vs V model

def plot_optimal_policies():
    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)

    seed = np.random.randint(0,1000)

    rewards, obstacles_map = init_map(10, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)

    neighbors = precompute_next_states(10, obstacles_map)

    start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)

    v = value_iteration(10, rewards, obstacles_map, gamma,neighbors)

    policy = extract_policy(v, obstacles_map,neighbors)

    agent_position = deepcopy(start)

    while agent_position!=goal:
        next_position = tuple(int(i) for i in policy[agent_position])
        visualize_policy_and_rewards(rewards, obstacles_map,policy)
        agent_position = next_position


    

    



if __name__ == "__main__":
    from dl_models import DeeperValueIterationModel,ValueIterationModel, UNet,UNetSmall
    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)

    # seeds = [281,88,31415,74767,12345,3,1999]
    seeds = np.random.randint(0,1000,25)


    # for seed in seeds:
    #     rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=seed)

    #     neighbors = precompute_next_states(n, obstacles_map)

    #     start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)
    #     # start = (0,0)
    #     # goal = (8,8)qq

    #     model = UNetSmall()
    #     model.load_state_dict(torch.load("model_weights/unet_small_5.pth",weights_only=True))

    #     # model = UNetSmall()
    #     # model.load_state_dict(torch.load("model_weights/unet_small_value_iteration_model.pth",weights_only=True))

    #     input = reformat_input(rewards, obstacles_map)
    #     input = input.unsqueeze(0)
    #     print(input.shape)

    #     V= model(input).detach().numpy().squeeze()

    #     Viter = value_iteration(n, rewards.copy(), obstacles_map, gamma,neighbors)
    #     V_init = nn_initialized_vi(model,n, rewards.copy(), obstacles_map, gamma,neighbors)

    #     path1 = get_nn_path(n, rewards.copy(), obstacles_map, neighbors, start, goal, model)
    #     path2 = get_vi_path(n, rewards.copy(), obstacles_map, neighbors, start, goal)
    #     path3 = get_vi_plus_nn_path(n, rewards.copy(), obstacles_map, neighbors, start, goal,model)

    #     vs = [V,Viter,V_init]
    #     paths = [path1,path2,path3]
    #     titles = ["Neural Network","Value Iteration","Value Iteration + Neural Network"]
    #     visually_compare_value_functions(vs, paths, rewards, obstacles_map, goal,seed=seed,titles=titles)


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

        paths = [path1,path2,path3,path4]

        compare_paths(paths, rewards, obstacles_map, goal,seed=seed)

  





    # plot_change_in_value()

    # rewards ,obstacles_map = init_map(10, config, num_blocks, num_obstacles, obstacle_type, square_size,obstacle_map=obstacles_map,seed=31415)
    # print(rewards)

    # print(obstacles_map)


    









