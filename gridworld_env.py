import numpy as np


class GridEnvironment:
    def __init__(self, n, rewards, obstacles, start, target):
        self.n = n
        self.initial_rewards = rewards.copy()
        self.rewards = rewards
        self.start = start
        self.target = target
        self.obstacles = obstacles
        # self.reset()

    def reset(self):
        self.current_position = self.start
        self.rewards = self.initial_rewards.copy()
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        return self.current_position,{}

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
    
        # Give a large reward if the agent reaches the target
        if done:
            reward += 10

        reward = reward - 0.1 # Add a small negative reward for each step to encourage shorter paths

        return self.current_position, reward, done, False, {}

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
    
    def get_cnn_input(self):
        pos = np.zeros((self.n,self.n))
        pos[self.current_position] = 1
        return np.stack([pos,self.rewards,self.obstacles],axis=0)
    

if __name__ == "__main__":
    # from utils import *
    # import pickle
    # from fo_solver import visualize_rewards

    # n, config, num_blocks, num_obstacles, obstacle_type, square_size, random_map, gamma = parse_arguments()

    # with open('obstacle.pkl', 'rb') as f:
    #     obstacles_map = pickle.load(f)

    # rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size, obstacle_map=obstacles_map)

    # start, goal = pick_start_and_goal(rewards, obstacles_map)

    # env = GridEnvironment(n, rewards, obstacles_map, start, goal)

    # obs, _ = env.reset()
    # done = False

    # t = env.get_cnn_input()
    # print(t.shape)
    pass

    
