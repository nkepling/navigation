import numpy as np
import matplotlib.pyplot as plt
from utils import *
import time
import pickle
from copy import deepcopy

# Define the input map
n = 50  # size of the grid
config = "block"  # distribution of positive probability cells
num_blocks = 3  # number of positive region blocks
num_obstacles = 3  # number of obstacles
obstacle_type = "block"
square_size = 4  # size of the positive region square

# Discount factor
gamma = 0.8

# define experiment configuration
random_map = True

def get_fov(cur_state, next_state, obstacles, map_shape, fov_range=10, fov_angle=60):
    """
    """
    def bresenham_line(start, end):
        """Bresenham's Line Algorithm to generate points between start and end."""
        x0, y0 = start
        x1, y1 = end
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    # Calculate the orientation vector and normalize it
    orientation = np.array(next_state) - np.array(cur_state)
    orientation = orientation / np.linalg.norm(orientation)

    # Generate the grid of cell coordinates around the current state
    x, y = np.meshgrid(
        np.arange(max(0, cur_state[0] - fov_range), min(map_shape[0], cur_state[0] + fov_range + 1)),
        np.arange(max(0, cur_state[1] - fov_range), min(map_shape[1], cur_state[1] + fov_range + 1))
    )
    grid_points = np.stack([x.ravel(), y.ravel()], axis=-1)

    # Calculate vectors from cur_state to each point in the grid
    vectors = grid_points - np.array(cur_state)

    # Compute distances to filter based on the range
    distances = np.linalg.norm(vectors, axis=1)
    in_range_mask = distances <= fov_range

    # Normalize the vectors
    unit_vectors = vectors / np.clip(distances[:, None], a_min=1e-8, a_max=None)

    # Compute dot products with the orientation vector
    dot_products = np.dot(unit_vectors, orientation)

    # Calculate angles and mask those within the FOV angle
    angles = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))
    in_fov_mask = angles <= fov_angle / 2

    # Combine range and FOV angle masks
    mask = in_range_mask & in_fov_mask

    # Apply mask to grid points
    fov_cells = grid_points[mask]

    # Filter cells using Bresenham's line algorithm to stop at obstacles
    visible_cells = []
    for cell in fov_cells:
        line = bresenham_line(cur_state, cell)
        # Check each point in the line for obstacles
        blocked = False
        for point in line:
            if obstacles[point[0], point[1]]:
                blocked = True
                break
        if not blocked:
            visible_cells.append(tuple(cell))

    return visible_cells



def get_fov_rewards(cur_sate,next_state,reward_array,obstacles):
    """Given the field of view array and the reward array, return the sum of rewards in the field of view.
    """
    fov_array = get_fov(cur_sate,next_state,obstacles,obstacles.shape)
    
    fov_array = np.array(fov_array)
    return np.sum(reward_array[fov_array[:,0],fov_array[:,1]])
    






def value_iteration_with_extened_fov(n, rewards, obstacles, gamma,neighbors,threshold=1e-6):
    """Value iteration with extended field of view (FOV)
    
    The agent can observe reward in a cone directly in front of it.

    Here you have to account for  the reward you get from each direction going into cell. 

    So rather than simply R(s,a) it is R(s,a,s'). 
    """

    V = np.zeros((n, n))
    loops = 0
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if obstacles[i, j]:
                    continue  # Skip obstacle cells
                v = V[i, j]
                # Get the possible next states and rewards
                next_states = []
                v_hat = []
                for s in neighbors[(i, j)]:
                    r = get_fov_rewards((i,j),s,rewards,obstacles)
                    v_hat.append(r + gamma * V[s[0], s[1]])

                # Bellman update
                #V[i, j] = rewards[i,j] + gamma * max(v_hat, default=0)
                V[i, j] = max(v_hat, default=0)
                delta = max(delta, abs(v - V[i, j])) # This works because we are looking for max difference

        loops += 1
        if delta < threshold:
            break
    return V

    


def finite_horizon_value_iteration(n, rewards, obstacles, neighbors, T,gamma=1):
    # Initialize the value function for all time steps
    V = np.zeros((T + 1, n, n))  # V[t] represents the value function at time t
    
    # Iterate backward in time from T-1 to 0
    for t in range(T - 1, -1, -1):  # Loop over time steps from T-1 to 0
        for i in range(n):
            for j in range(n):
                if obstacles[i, j]:
                    continue  # Skip obstacle cells
                # Get the possible next states and rewards
                v_hat = []
                for s in neighbors[(i, j)]:
                    v_hat.append(V[t + 1, s[0], s[1]])  # Use value function at time t+1

                # Bellman update for time step t
                V[t, i, j] = rewards[i, j] + gamma * max(v_hat, default=0)

    # Return the value function at time 0 (V[0]) and all time steps if needed
    return V[0], V


# Function for Value Iteration
def value_iteration(n, rewards, obstacles, gamma,neighbors,threshold=1e-6):
    # Initialize the value function
    V = np.zeros((n, n))
    loops = 0
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if obstacles[i, j]:
                    continue  # Skip obstacle cells
                v = V[i, j]
                # Get the possible next states and rewards
                next_states = []
                v_hat = []
                for s in neighbors[(i, j)]:
                    v_hat.append(V[s[0], s[1]])

                # Bellman update
                V[i, j] = rewards[i, j] + gamma * max(v_hat, default=0)
                delta = max(delta, abs(v - V[i, j])) # This works because we are looking for max difference

        loops += 1
        if delta < threshold:
            break
    return V


# Function to extract the policy
def extract_policy(V, obstacles,neighbors,n):
    policy = np.zeros((n, n, 2))

    for i in range(n):
        for j in range(n):
            if obstacles[i, j]:
                continue  # Skip obstacle cells
            # Get the possible next states and rewards
            v_hat = {}
            for s in neighbors[(i, j)]:
                v_hat[(s[0], s[1])] = V[s[0], s[1]]
            # Choose the action that maximizes the value
            if v_hat:
                best_next_state = max(v_hat, key=v_hat.get)
                policy[i, j] = best_next_state

    return policy

def policy_iteration(rewards, obstacles, next_states, gamma,threshold=1e-6):
    n = rewards.shape[0]

    # Initialize policy randomly
    policy = np.zeros((n, n, 2), dtype=int)
    for i in range(n):
        for j in range(n):
            if not obstacles[i, j]:
                policy[i, j] = [i, j]  # Initial policy: stay in place (self-transition)

    # Initialize value function
    V = np.zeros((n, n))

    def policy_evaluation(policy, V, rewards, obstacles, gamma, threshold):
        while True:
            delta = 0
            for i in range(n):
                for j in range(n):
                    if obstacles[i, j]:
                        continue  # Skip obstacle cells
                    v = V[i, j]
                    next_i, next_j = policy[i, j]
                    V[i, j] = rewards[i, j] + gamma * V[next_i, next_j]
                    delta = max(delta, abs(v - V[i, j]))
            if delta < threshold:
                break
        return V

    def policy_improvement(V, policy, rewards, obstacles, next_states, gamma):
        policy_stable = True
        for i in range(n):
            for j in range(n):
                if obstacles[i, j]:
                    continue  # Skip obstacle cells
                old_action = policy[i, j]

                # Find the best action
                best_value = float('-inf')
                best_action = old_action
                for next_i, next_j in next_states[(i, j)]:
                    value = rewards[i, j] + gamma * V[next_i, next_j]
                    if value > best_value:
                        best_value = value
                        best_action = (next_i, next_j)

                policy[i, j] = best_action

                if not np.array_equal(old_action, policy[i, j]):
                    policy_stable = False
        return policy, policy_stable

    while True:
        V = policy_evaluation(policy, V, rewards, obstacles, gamma, threshold)
        policy, policy_stable = policy_improvement(V, policy, rewards, obstacles, next_states, gamma)
        if policy_stable:
            break

    return V, policy

# Function to visualize policy with rewards and obstacles
def visualize_policy_and_rewards(rewards, obstacles, policy):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the reward function
    display_matrix = np.copy(rewards)
    display_matrix[obstacles] = np.nan  # Set obstacles to NaN for black color
    im = ax.imshow(display_matrix, cmap='viridis', origin='upper')
    ax.set_title("Policy with Rewards and Obstacles in Background")
    fig.colorbar(im, ax=ax)

    # Plot the policy arrows
    for i in range(n):
        for j in range(n):
            if obstacles[i, j]:
                continue  # Skip obstacle cells
            if policy[i, j][0] < i:
                ax.arrow(j, i, 0, -0.5, head_width=0.2, head_length=0.2, fc='r', ec='r')
            if policy[i, j][0] > i:
                ax.arrow(j, i, 0, 0.5, head_width=0.2, head_length=0.2, fc='r', ec='r')
            if policy[i, j][1] < j:
                ax.arrow(j, i, -0.5, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
            if policy[i, j][1] > j:
                ax.arrow(j, i, 0.5, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')

    ax.invert_yaxis()
    ax.grid(True)
    plt.show()

# Function to visualize policy with rewards and obstacles
def visualize_rewards(rewards, obstacles, start, goal, curr_pos=None, next_pos=None):
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

    ax.invert_yaxis()
    ax.grid(True)
    plt.show()


def visualize_values(value_map, obstacles, start, goal, curr_pos=None):
    """
    Visualize the value map with obstacles, start, goal, and current position (without arrows).
    
    Args:
        value_map (np.ndarray): 2D array representing the value of each grid cell.
        obstacles (np.ndarray): 2D binary array where 1 indicates an obstacle.
        start (tuple): Coordinates of the start position (row, col).
        goal (tuple): Coordinates of the goal position (row, col).
        curr_pos (tuple, optional): Coordinates of the current position (row, col). Default is None.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Prepare the display matrix with NaN for obstacles
    display_matrix = np.copy(value_map)
    display_matrix[obstacles] = np.nan  # Set obstacles to NaN for black color

    # Plot the value map with viridis colormap
    im = ax.imshow(display_matrix, cmap='viridis', origin='upper')
    ax.set_title("Value Map with Obstacles, Start, Goal, and Agent")
    fig.colorbar(im, ax=ax)

    # Plot start and goal positions
    ax.plot(start[1], start[0], 'bo', label='Start')  # Start in blue
    ax.plot(goal[1], goal[0], 'go', label='Goal')    # Goal in green

    # Plot current position if provided
    if curr_pos is not None:
        ax.plot(curr_pos[1], curr_pos[0], 'ro', markersize=10, label='Current')  # Current position in red

    # Adjust plot settings
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    import torch


    if not random_map:
        with open('mdp_data.pkl', 'rb') as f:
            rewards, obstacles_map, neighbors = pickle.load(f)
    else:
        # Initialize the reward matrix: directly reward based on the probability density of target location
        rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)
        neighbors = precompute_next_states(n, obstacles_map)
        # Save the data to a file using pickle
        with open('mdp_data.pkl', 'wb') as f:
            pickle.dump((rewards, obstacles_map, neighbors), f)
            
    print("Obstacle Map:")
    print(obstacles_map.shape)
    # print("Rewards:", rewards)
    """ Simple Value Iteration"""
    # start_time = time.time()
    # V = value_iteration(n, rewards, obstacles_map, gamma)
    # policy = extract_policy(V, obstacles_map)
    # end_time = time.time()
    # print("Total time taken for VI: {} seconds".format(end_time - start_time))
    # # Visualize the optimal policy with rewards in the background
    # visualize_policy_and_rewards(rewards, obstacles_map, policy)

    """ Navigation """
    start, goal = pick_start_and_goal(rewards, obstacles_map)
    visualize_rewards(rewards, obstacles_map, start, goal)
    if start == goal:
        print("the agent is already in the target position")

    agent_position = deepcopy(start)
    while agent_position!=goal:
        # mark current position as 0 reward
        rewards[agent_position[0], agent_position[1]] = 0
        # V = value_iteration(n, rewards, obstacles_map, gamma,neighbors)
        V = value_iteration_with_extened_fov(n,rewards,obstacles_map,gamma,neighbors)
        policy = extract_policy(V, obstacles_map,neighbors)
        next_position = tuple(int(i) for i in policy[agent_position])
        print("Agent next state is {}".format(next_position))
        i, j = agent_position[0], agent_position[1]
        # visualize_rewards(rewards, obstacles_map, start, goal, agent_qposition, next_position)
        visualize_policy_and_rewards(rewards, obstacles_map, policy)
        agent_position = next_position









