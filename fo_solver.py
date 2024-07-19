import numpy as np
import matplotlib.pyplot as plt
from utils import *
import time
import pickle
from copy import deepcopy

# Define the input map
n = 10  # size of the grid
config = "block"  # distribution of positive probability cells
num_blocks = 3  # number of positive region blocks
num_obstacles = 3  # number of obstacles
obstacle_type = "block"
square_size = 4  # size of the positive region square

# Discount factor
gamma = 0.8

# define experiment configuration
random_map = True

# Function for Value Iteration
def value_iteration(n, rewards, obstacles, gamma, threshold=1e-6):
    # Initialize the value function
    V = np.zeros((n, n))
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
                delta = max(delta, abs(v - V[i, j]))
        if delta < threshold:
            break
    return V

# Function to extract the policy
def extract_policy(V, obstacles):
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

def policy_iteration(rewards, obstacles, next_states, gamma, threshold=1e-6):
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


if __name__ == "__main__":
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
        V = value_iteration(n, rewards, obstacles_map, gamma)
        policy = extract_policy(V, obstacles_map)
        next_position = tuple(int(i) for i in policy[agent_position])
        print("Agent next state is {}".format(next_position))
        i, j = agent_position[0], agent_position[1]
        visualize_rewards(rewards, obstacles_map, start, goal, agent_position, next_position)
        agent_position = next_position









