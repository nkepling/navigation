import random
import numpy as np
from utils import *
from cnn import *
from fo_solver import visualize_rewards
# Visualize the path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hashlib
import time


class GridEnvironment:
    def __init__(self, n, rewards, obstacles, start, target):
        self.n = n
        self.initial_rewards = rewards.copy()
        self.rewards = rewards
        self.start = start
        self.target = target
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.current_position = self.start
        self.rewards = self.initial_rewards.copy()
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        return self.current_position, self.visited.copy()

    def step(self, action):
        i, j = self.current_position
        next_position = (i, j)

        if action == 0 and i > 0:  # Move up
            next_position = (i - 1, j)
        elif action == 1 and i < self.n - 1:  # Move down
            next_position = (i + 1, j)
        elif action == 2 and j > 0:  # Move left
            next_position = (i, j - 1)
        elif action == 3 and j < self.n - 1:  # Move right
            next_position = (i, j + 1)

        # Check if the next position is an obstacle
        if not self.obstacles[next_position]:
            self.current_position = next_position

        # Collect reward only if the cell is visited for the first time
        if not self.visited[self.current_position]:
            reward = self.rewards[self.current_position]
            self.visited[self.current_position] = True
            self.rewards[self.current_position] = 0  # Set reward to 0 after collecting
        else:
            reward = 0

        done = self.current_position == self.target

        return (self.current_position, self.visited.copy()), reward, done

    def get_valid_actions(self):
        i, j = self.current_position
        actions = []

        if i > 0 and not self.obstacles[i - 1, j]:  # Move up
            actions.append(0)
        if i < self.n - 1 and not self.obstacles[i + 1, j]:  # Move down
            actions.append(1)
        if j > 0 and not self.obstacles[i, j - 1]:  # Move left
            actions.append(2)
        if j < self.n - 1 and not self.obstacles[i, j + 1]:  # Move right
            actions.append(3)

        return actions

    def get_state_space_size(self):
        return self.n * self.n * 2 ** (self.n * self.n)

    def get_action_space_size(self):
        return 4  # Up, Down, Left, Right
    
    def get_updated_rewards(self):
        return self.rewards

class DQLearningAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}

    def get_state_index(self, position, visited):
        visited_str = ''.join(map(str, visited.flatten().astype(int)))
        state_str = f"{position}_{visited_str}"
        return hashlib.md5(state_str.encode()).hexdigest()

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return random.choice(valid_actions)
        else:
            state_q_values = self.q_table[state]
            max_q_value = max(state_q_values.get(a, 0) for a in valid_actions)
            max_q_actions = [a for a in valid_actions if state_q_values.get(a, 0) == max_q_value]
            return random.choice(max_q_actions)

    def update(self, state, action, reward, next_state, valid_actions):
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        best_next_action = max(valid_actions, key=lambda a: self.q_table[next_state].get(a, 0))
        td_target = reward + self.gamma * self.q_table[next_state].get(best_next_action, 0)
        td_error = td_target - self.q_table[state].get(action, 0)
        self.q_table[state][action] = self.q_table[state].get(action, 0) + self.alpha * td_error

# def visualize_path(path, rewards, obstacle_map, target):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     display_matrix = np.copy(rewards)
#     display_matrix[obstacle_map == 0] = -1  # Set obstacles to a special value
#
#     cmap = plt.cm.viridis
#     cmap.set_bad(color='red')
#     cmap.set_under(color='red')
#     bounds = np.linspace(0, 1, 256)
#     new_cmap = mcolors.ListedColormap(cmap(bounds))
#
#     # masked_display_matrix = np.ma.masked_where(display_matrix == -1, display_matrix)
#     ax.imshow(display_matrix, cmap=new_cmap, origin='upper')
#     ax.plot(target[1], target[0], 'go')  # Target position in green
#
#     for position in path:
#         ax.plot(position[1], position[0], 'bo')  # Agent's path in blue
#
#     ax.invert_yaxis()
#     ax.grid(True)
#     plt.show()


# Define the input map
n = 10  # size of the grid
config = "block"  # distribution of positive probability cells
num_blocks = 3  # number of positive region blocks
num_obstacles = 3  # number of obstacles
obstacle_type = "block"
square_size = 4  # size of the positive region square
rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)
neighbors = precompute_next_states(n, obstacles_map)
start, target = pick_start_and_goal(rewards, obstacles_map)
visualize_rewards(rewards, obstacles_map, start, target)

# Create the environment
env = GridEnvironment(n, rewards, obstacles_map, start, target)
# Create the Q-learning agent
state_space_size = env.get_state_space_size()
action_space_size = env.get_action_space_size()
agent = DQLearningAgent(state_space_size, action_space_size)

# Training loop
num_episodes = 10000
max_steps_per_episode = 100

start_time = time.time()
for episode in range(num_episodes):
    position, visited = env.reset()
    state = agent.get_state_index(position, visited) #FIXME we want the state to be a matrix of the position and visited cells not a index
    total_reward = 0

    for step in range(max_steps_per_episode):
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)
        (next_position, next_visited), reward, done = env.step(action)
        next_state = agent.get_state_index(next_position, next_visited)

        agent.update(state, action, reward, next_state, valid_actions)

        position, visited = next_position, next_visited
        state = next_state
        total_reward += reward

        if done:
            break

    # if (episode + 1) % 100 == 0:
    #     print(f"Episode {episode + 1}, Total Reward: {total_reward}")
end_time = time.time()
print("Training completed in {} seconds".format(end_time-start_time))

# Evaluation
position, visited = env.reset()
state = agent.get_state_index(position, visited)
total_reward = 0
path = [env.current_position]

for step in range(max_steps_per_episode):
    valid_actions = env.get_valid_actions()
    start_time = time.time()
    action = agent.choose_action(state, valid_actions)
    end_time = time.time()
    print("Inference completed in {} seconds".format(end_time - start_time))
    (next_position, next_visited), reward, done = env.step(action)
    next_state = agent.get_state_index(next_position, next_visited)

    position, visited = next_position, next_visited
    state = next_state
    total_reward += reward
    path.append(next_position)


    if done:
        break

print(f"Evaluation complete. Total Reward: {total_reward}")
print(f"Path taken: {path}")


# Visualization
def visualize_path(path, rewards, target, obstacles):
    fig, ax = plt.subplots(figsize=(10, 10))
    display_matrix = np.copy(rewards)
    display_matrix[obstacles] = np.nan  # Set obstacles to NaN for black color

    cmap = plt.cm.viridis
    cmap.set_bad(color='red')
    cmap.set_under(color='red')
    bounds = np.linspace(0, 1, 256)
    new_cmap = mcolors.ListedColormap(cmap(bounds))

    masked_display_matrix = np.ma.masked_where(display_matrix == -1, display_matrix)
    ax.imshow(masked_display_matrix, cmap=new_cmap, origin='upper')
    ax.plot(target[1], target[0], 'go')  # Target position in green

    # Plot arrows for the agent's path
    for k in range(len(path) - 1):
        start_i, start_j = path[k]
        end_i, end_j = path[k + 1]
        dx = end_j - start_j
        dy = end_i - start_i
        ax.arrow(start_j, start_i, dx, dy, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
    ax.invert_yaxis()
    ax.grid(True)
    plt.show()


visualize_path(path, rewards, target, obstacles_map)

