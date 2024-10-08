"""
Utility File
"""
import numpy as np
import random
import yaml

import torch
"""
Takes config: single double random
and size n as inputs
"""

# n = None
# config = None
# num_blocks = None
# num_obstacles = None
# obstacle_type = None
# square_size = None
# random_map = None
# gamma = None





def init_map(n, config, num_blocks, num_obstacles, obstacle_type="block", square_size=10,obstacle_map=None,seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    rewards = np.zeros((n, n))
    obstacles_map = np.zeros((n, n), dtype=bool)

    if config == "block":
        for _ in range(num_blocks):
            # Randomly sample start_x and start_y
            start_x = random.randrange(0, n)
            start_y = random.randrange(0, n)

            # Calculate the end coordinates, ensuring they do not go out of bounds
            end_x = min(start_x + square_size, n)
            end_y = min(start_y + square_size, n)

            # Place positive rewards between 1 and 10 in the defined square
            rewards[start_x:end_x, start_y:end_y] = np.random.randint(1, 100, (end_x - start_x, end_y - start_y),)
            # rewards[start_x:end_x, start_y:end_y] = np.random.randint(1, 2, (end_x - start_x, end_y - start_y))

    
    if obstacle_map is not None:
        obstacles_map = obstacle_map
        rewards[obstacles_map] = 0

    elif num_obstacles > 0:
        obstacle_square_size = 4
        if obstacle_type == "random":  # Randomly place obstacles
            for _ in range(num_obstacles):
                x, y = np.random.randint(0, n, size=2)
                if rewards[x, y] != 0:  # Ensure obstacle isn't in the reward region
                    rewards[x, y] = 0
                obstacles_map[x, y] = True

        elif obstacle_type == "block":
            for _ in range(num_obstacles):
                start_x = random.randrange(0, n)
                start_y = random.randrange(0, n)

                # Calculate the end coordinates, ensuring they do not go out of bounds
                end_x = min(start_x + obstacle_square_size, n)
                end_y = min(start_y + obstacle_square_size, n)

                rewards[start_x:end_x, start_y:end_y] = 0
                obstacles_map[start_x:end_x, start_y:end_y] = True
    #
    # # test blocks
    # rewards[0, 0] = 0.01
    # rewards[5, 0] = 0.01

    # Normalize rewards to sum to 1
    total_sum = np.sum(rewards)
    if total_sum != 0:
        rewards = rewards / total_sum

    return rewards, obstacles_map


import numpy as np
import random
from collections import deque

def is_reachable(rewards, obstacles_map):
    """
    Check if all non-obstacle cells are reachable from the first non-obstacle cell using BFS.
    """
    n = rewards.shape[0]
    visited = np.zeros_like(obstacles_map, dtype=bool)

    # Find the first non-obstacle cell
    start = None
    for i in range(n):
        for j in range(n):
            if not obstacles_map[i, j]:  # A non-obstacle cell
                start = (i, j)
                break
        if start:
            break

    if start is None:
        return True  # If there are no non-obstacle cells, we consider it trivially valid.

    # Perform BFS to mark all reachable cells
    queue = deque([start])
    visited[start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not obstacles_map[nx, ny] and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append((nx, ny))

    # Check if all non-obstacle cells have been visited
    return np.all(visited | obstacles_map)


def init_reachable_map(n, config, num_blocks, num_obstacles, obstacle_type="block", square_size=10, obstacle_map=None, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
        
    while True:  # Loop until we generate a valid map
        rewards = np.zeros((n, n))
        obstacles_map = np.zeros((n, n), dtype=bool)

        if config == "block":
            for _ in range(num_blocks):
                start_x = random.randrange(0, n)
                start_y = random.randrange(0, n)
                end_x = min(start_x + square_size, n)
                end_y = min(start_y + square_size, n)
                rewards[start_x:end_x, start_y:end_y] = np.random.randint(1, 100, (end_x - start_x, end_y - start_y))

        if obstacle_map is not None:
            obstacles_map = obstacle_map
            rewards[obstacles_map] = 0

        elif num_obstacles > 0:
            obstacle_square_size = 4
            if obstacle_type == "random":
                for _ in range(num_obstacles):
                    x, y = np.random.randint(0, n, size=2)
                    if rewards[x, y] != 0:
                        rewards[x, y] = 0
                    obstacles_map[x, y] = True

            elif obstacle_type == "block":
                for _ in range(num_obstacles):
                    start_x = random.randrange(0, n)
                    start_y = random.randrange(0, n)
                    end_x = min(start_x + obstacle_square_size, n)
                    end_y = min(start_y + obstacle_square_size, n)
                    rewards[start_x:end_x, start_y:end_y] = 0
                    obstacles_map[start_x:end_x, start_y:end_y] = True

        # Normalize rewards to sum to 1
        total_sum = np.sum(rewards)
        if total_sum != 0:
            rewards = rewards / total_sum

        # Check if all non-obstacle cells are reachable
        if is_reachable(rewards, obstacles_map):
            break  # If the map is valid, exit the loop and return the result
        else:
            print("Regenerating map: Non-obstacle cells are not fully reachable.")

    return rewards, obstacles_map


def init_random_reachable_map(n, config, num_blocks, min_obstacles, max_obstacles, obstacle_type="block", square_size=10, obstacle_map=None, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
        
    while True:  # Loop until we generate a valid map
        rewards = np.zeros((n, n))
        obstacles_map = np.zeros((n, n), dtype=bool)

        if config == "block":
            for _ in range(num_blocks):
                start_x = random.randrange(0, n)
                start_y = random.randrange(0, n)
                end_x = min(start_x + square_size, n)
                end_y = min(start_y + square_size, n)
                rewards[start_x:end_x, start_y:end_y] = np.random.randint(1, 100, (end_x - start_x, end_y - start_y))

        if obstacle_map is not None:
            obstacles_map = obstacle_map
            rewards[obstacles_map] = 0

        else:
            # Randomize the number of obstacles within a given range
            num_obstacles = random.randint(min_obstacles, max_obstacles)

            # Define obstacle diversity parameters
            obstacle_square_size = random.randint(1, 5)  # Randomize obstacle size
            # obstacle_square_size = random.randint(1,2)
            obstacle_cluster_prob = 0.3  # Probability of forming obstacle clusters
            
            if obstacle_type == "random":
                for _ in range(num_obstacles):
                    x, y = np.random.randint(0, n, size=2)
                    if rewards[x, y] != 0:
                        rewards[x, y] = 0
                    obstacles_map[x, y] = True

            elif obstacle_type == "block":
                for _ in range(num_obstacles):
                    start_x = random.randrange(0, n)
                    start_y = random.randrange(0, n)
                    end_x = min(start_x + obstacle_square_size, n)
                    end_y = min(start_y + obstacle_square_size, n)
                    rewards[start_x:end_x, start_y:end_y] = 0
                    obstacles_map[start_x:end_x, start_y:end_y] = True
                    
                    # Randomly create obstacle clusters
                    if random.random() < obstacle_cluster_prob:
                        cluster_size = random.randint(1, 3)  # Size of the cluster
                        for _ in range(cluster_size):
                            cx = min(start_x + random.randint(-2, 2), n-1)
                            cy = min(start_y + random.randint(-2, 2), n-1)
                            if 0 <= cx < n and 0 <= cy < n:
                                obstacles_map[cx, cy] = True
                                rewards[cx, cy] = 0

        # Normalize rewards to sum to 1
        total_sum = np.sum(rewards)
        if total_sum != 0:
            rewards = rewards / total_sum

        # # Check if all non-obstacle cells are reachable
        # if is_reachable(rewards, obstacles_map):
        #     break  # If the map is valid, exit the loop and return the result
        # else:
        #     print("Regenerating map: Non-obstacle cells are not fully reachable.")

        # Check that the agent is not in an obstacle cell

        if obstacles_map[0, 0]:
            print("Regenerating map: Agent is in an obstacle cell.")
        else:
            break

    return rewards, obstacles_map






"""
Precompute the set of adjacent states for each state
"""
def precompute_next_states(n, obstacles):
    next_states = {}
    for i in range(n):
        for j in range(n):
            if obstacles[i, j]:
                continue
            next_states[(i, j)] = []
            if i > 0 and not obstacles[i-1, j]:
                next_states[(i, j)].append((i-1, j))
            if i < n-1 and not obstacles[i+1, j]:
                next_states[(i, j)].append((i+1, j))
            if j > 0 and not obstacles[i, j-1]:
                next_states[(i, j)].append((i, j-1))
            if j < n-1 and not obstacles[i, j+1]:
                next_states[(i, j)].append((i, j+1))
    return next_states


import random

# randomly pick a start and goal positions
# Sample a target from the positive reward cells
def pick_start_and_goal(rewards, obstacles,seed=None):
    # positive_reward_cells = [(i, j) for (i, j), reward in np.ndenumerate(rewards) if reward > 0]
    # target = random.choice(positive_reward_cells)
    # print(f"Target: {target}")
    # weighted sampling based on the probability of finding the target
    flat_rewards = rewards.flatten()
    total_sum = np.sum(flat_rewards)
    
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    if total_sum == 0:
        raise ValueError("No positive rewards to sample from.")

    probabilities = flat_rewards / total_sum
    while True:
        goal_index = np.random.choice(np.arange(len(flat_rewards)), p=probabilities)
        target = np.unravel_index(goal_index, rewards.shape)
        target = tuple([int(i) for i in target])
        if not obstacles[target] or target == (0, 0):
            break
    
   

    # randomly sample start position
    start = (0, 0)
    # print(f"Start: {start}")
    # print(f"Target: {target}")
    return start, target


def load_env_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_and_initialize():
    global n, config, num_blocks, num_obstacles, obstacle_type, square_size, random_map, gamma
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", type=str, default="env_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    env_config = load_env_config(args.env_config)
    
    # Initialize the global constants
    n = env_config["n"]
    config = env_config["config"]
    num_blocks = env_config["num_blocks"]
    num_obstacles = env_config["num_obstacles"]
    obstacle_type = env_config["obstacle_type"]
    square_size = env_config["square_size"]
    random_map = env_config["random_map"]
    gamma = env_config["gamma"]

# Function to ensure constants are initialized when needed
def ensure_initialized():
    if n is None or config is None:
        parse_and_initialize()

def format_input_for_ddqn_cnn(state):
    if not isinstance(state,torch.Tensor):
        state = torch.tensor(state)
        state = state.float()
    
    state = state.unsqueeze(0)
    assert(state.shape == (1,3,10,10))

    return state

if __name__ == "__main__":
    ensure_initialized()
    print(f"n = {n}, config = {config}, num_blocks = {num_blocks}, gamma = {gamma}")


    
    




