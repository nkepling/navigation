import numpy as np
import torch

from fo_solver import precompute_next_states
from fo_solver import init_random_reachable_map
from fo_solver import pick_start_and_goal


class GridEnvironment:
    def __init__(self, config, rewards, obstacles, start, target, living_reward=None, shuffle=False,train=True,max_steps=1000):
        self.n = config.n
        self.config = config
        self.shuffle = shuffle
        self.initial_rewards = rewards.copy()
        self.rewards = rewards
        self.start = start
        self.target = target
        self.obstacles = obstacles
        self.living_reward = living_reward
        self.current_position = self.start
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        self.train = train
        
        self.neighbors = precompute_next_states(self.n, obstacles)
        self.action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        self.max_steps = max_steps

    def reset(self, coords=None, rewards=None):
        # Generate new maps and positions if shuffling is enabled
        self.step_count = 0
        if self.shuffle:
            rewards, obstacles_map = init_random_reachable_map(
                self.config.n,
                self.config.obstacle_shape,
                self.config.num_obstacles,
                self.config.min_obstacles,
                self.config.max_obstacles,
                obstacle_type=self.config.obstacle_type,
                square_size=self.config.square_size,
                obstacle_map=self.config.obstacle_map,
                seed=self.config.seed,
                num_reward_blocks=self.config.num_reward_blocks,
                reward_square_size=self.config.reward_square_size,
                obstacle_cluster_prob=self.config.obstacle_cluster_prob,
                obstacle_square_sizes=self.config.obstacle_square_sizes
            )
            
            start, target = pick_start_and_goal(rewards, obstacles_map)
            self.rewards = rewards
            self.obstacles = obstacles_map
            self.start = start
            self.target = target
            self.neighbors = precompute_next_states(self.n, obstacles_map)
        else:
            # Reset to initial conditions if shuffling is not enabled
            self.rewards = self.initial_rewards.copy()
            self.current_position = self.start
            self.visited = np.zeros((self.n, self.n), dtype=bool)

        return self.current_position, {}

    def step(self, action):
        i, j = self.current_position
        direction = self.action_to_dir[action]
        next_position = (i + direction[0], j + direction[1])

        # Check if the next position is an obstacle
        if not self.obstacles[next_position]:
            self.current_position = next_position
            self.step_count += 1
        else:
            raise ValueError("Invalid action: Next position is an obstacle")

        # Collect reward only if the cell is visited for the first time
        if self.visited[self.current_position]:
            reward = self.rewards[self.current_position]
            self.visited[self.current_position] = True
            self.rewards[self.current_position] = 0  # Set reward to 0 after collecting
        else:
            reward = 0

        if self.living_reward:
            reward += self.living_reward

        if self.train:
            done = self.step_count >= self.max_steps
        else: # test
            done = (self.current_position == self.target)

        return self.current_position, reward, done, False, {}

    def get_valid_actions(self, state=None):
        if state is None:
            i, j = self.current_position
        else:
            i, j = state

        actions = []
        for action, (di, dj) in self.action_to_dir.items():
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.n and 0 <= new_j < self.n and not self.obstacles[new_i, new_j]:
                actions.append(action)

        return np.array(actions)
    
    def get_valid_next_states(self, state=None):
        """Return dict of action actions that lead to valid next states.
        """
        if state is None:
            i, j = self.current_position
        else:
            i, j = state

        next_states = {}
        for action, (di, dj) in self.action_to_dir.items():
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.n and 0 <= new_j < self.n and not self.obstacles[new_i, new_j]:
                next_states[action] = (new_i, new_j)

        return next_states

    def get_vin_input(self, state=None):
        """Return input for the VIN model."""
        if not state:
            state_x, state_y = self.current_position
        else:
            state_x, state_y = state

        state_x = torch.tensor(state_x)
        state_y = torch.tensor(state_y)
    
        rewards = torch.tensor(self.get_rewards(), dtype=torch.float32).unsqueeze(0)
        obstacles_map = torch.tensor(self.obstacles, dtype=torch.float32).unsqueeze(0)
        input = torch.cat((rewards, obstacles_map), dim=0).unsqueeze(0)

        return input, state_x, state_y

    def get_rewards(self):


        return self.rewards
    




    

if __name__ == "__main__":
    pass

    
