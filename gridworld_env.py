import numpy as np
import torch
from nn_training import reformat_input, reformat_output
from fo_solver import precompute_next_states


class GridEnvironment:
    def __init__(self, n, rewards, obstacles, start, target):
        self.n = n
        self.initial_rewards = rewards.copy()
        self.rewards = rewards
        self.start = start
        self.target = target
        self.obstacles = obstacles

        self.neighbors = precompute_next_states(n, obstacles) # Precompute the neighbors for each state : dict 

        self.action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)} # up, right , down, left

        # self.reset()

    def reset(self,coords=None):
        self.current_position = self.start
        self.rewards = self.initial_rewards.copy()
        self.visited = np.zeros((self.n, self.n), dtype=bool)
        return self.current_position,{}

    def step(self, action):
        i, j = self.current_position
        #next_position = (i, j)

        #action_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
        #action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        dir = self.action_to_dir[action]
        next_position = (i + dir[0], j + dir[1])

        # Check if the next position is an obstacle
        if not self.obstacles[next_position]:
            self.current_position = next_position
        else: 
            raise ValueError("Invalid action: Next position is an obstacle")

        # Collect reward only if the cell is visited for the first time
        if not self.visited[self.current_position]:
            reward = self.rewards[self.current_position]
            self.visited[self.current_position] = True
            self.rewards[self.current_position] = 0  # Set reward to 0 after collecting
        else:
            reward = 0

        done = False
        # Give a large reward if the agent reaches the target
        # if done:
        #     reward += 1

        return self.current_position, reward, done, False, {}
    
    def get_valid_actions(self, state=None):
        if state is None:
            i, j = self.current_position
        else:
            i, j = state

        actions = []

        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Check each action for validity
        for action, (di, dj) in enumerate(directions):
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.n and 0 <= new_j < self.n and not self.obstacles[new_i, new_j]:
                actions.append(action)

        return np.array(actions)


    def get_state_space_size(self):
        return self.n * self.n * 2 ** (self.n * self.n)

    def get_action_space_size(self):
        return 4  # Up, Down, Left, Right
    
    def get_rewards(self):
        return self.rewards
    
    def get_vin_input(self,state=None):
        """Return input for the VIN model
        """

        if not state:
            state_x, state_y = self.current_position
        else:
            state_x, state_y = state

        state_x = torch.tensor(state_x)
        state_y = torch.tensor(state_y)
    
        rewards = self.get_rewards()
        obstacles = self.obstacles

        temp = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
        # obstacles_map  = np.where(obstacles_map,-1,0)
        obstacles_map = torch.tensor(obstacles, dtype=torch.float32).unsqueeze(0)
        input = torch.cat((temp, obstacles_map), dim=0).unsqueeze(0)

        return input, state_x, state_y
    
    def get_valid_next_states(self, state=None):
        if state is None:
            state = self.current_position
            i, j = state    
        else:
            i, j = state

        valid_actions = self.get_valid_actions(state)
        next_states = {}
        for action in valid_actions:
            dir = self.action_to_dir[action]
            next_position = (i + dir[0], j + dir[1])
            next_states[action] = next_position

        return next_states
    








    

if __name__ == "__main__":
    pass

    
