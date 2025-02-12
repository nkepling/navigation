import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import hashlib
from fo_solver import visualize_rewards
"""Environment class for problem gridworld.

states: (rewards, obstacles, agent_position)
actions: 0 (left), 1 (down), 2 (right), 3 (up)
rewards: The rewards at each cell in the grid.
obstacles: A binary grid where 1 indicates an obstacle.
transitions: If the agent inta positive reward cell and does not find the goal, the reward is set to 0. 
"""

class GridworldEnv(gym.Env):
    def __init__(self, rewards, obstacles,start_pos=(0,0),goal_pos=(1,1),living_reward=0.0):
        """
        Initialize the Gridworld environment.
        
        Args:
        - rewards: A numpy array representing the rewards at each cell.
        - obstacles: A numpy array of the same shape as rewards, where 1 indicates an obstacle and 0 otherwise.
        - collision_penalty: The penalty for colliding with an obstacle.
        """
        super(GridworldEnv, self).__init__()

        assert rewards.shape == obstacles.shape, "Rewards and obstacles must have the same shape."
        self.rewards = rewards.copy()
        self.obstacles = obstacles.copy()
        self.starting_rewards = rewards.copy()
        self.action_to_dir = {0: np.array([0, -1]), 1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([-1, 0])}
        self.starting_position = np.array([0, 0])
        self.agent_position = np.array(start_pos)
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.living_reward = living_reward

        self.height, self.width = rewards.shape

        self.observation_space = gym.spaces.Dict({
            "rewards": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.height, self.width), dtype=np.float32),
            "obstacles": gym.spaces.MultiBinary((self.height, self.width)),
            "agent_position": gym.spaces.MultiDiscrete([self.height, self.width])
        })

        self.collection_bonus = 1

        self.action_space = gym.spaces.Discrete(4)  # 4 actions: left, down, right, up

    def reset(self,seed=None,options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        self.agent_position = np.array([0,0]) # Agent starts in the top-left corner
        self.current_rewards = self.starting_rewards.copy()
        self.terminated = False
        self.truncated = False  
        return self.get_state(), {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
        - action: One of 0 (left), 1 (down), 2 (right), 3 (up).
        
        Returns:
        - observation: Dictionary containing the new state
        - reward: Reward for this step
        - terminated: Whether the episode ends due to reaching terminal state
        - truncated: Whether the episode was stopped due to other factors
        - info: Additional debug info
        """
        dx, dy = self.action_to_dir[action]
        x, y = self.agent_position
        new_x, new_y = x + dx, y + dy

        info = {}
        info["collision"] = False

        # self.agent_position = np.array([new_x, new_y])  # Update agent's position

        # Check for out-of-bounds movement
        if new_x < 0 or new_x >= self.height or new_y < 0 or new_y >= self.width:
            reward = -0.1 # Penalty for hitting boundary
            reward += self.living_reward
            self.terminated = False
            state = self.get_state()
            info["collision"] = True
            
            

        # Check for obstacles
        elif self.obstacles[new_x, new_y] == 1:
            reward = -0.1  # Penalty for hitting obstacle
            reward += self.living_reward
            self.terminated = False
            info["collision"] = True
            state = self.get_state()
        
        # Collect reward 
        else:
            reward = self.current_rewards[new_x, new_y]

            state = self.get_state()

            if reward > 0:
                self.current_rewards[new_x, new_y] = 0

            reward += self.living_reward
            self.agent_position = np.array([new_x, new_y])  
            self.terminated = False

            if np.all(self.current_rewards == 0):
                self.terminated = True
                # reward += 10 
            
        return state, reward, self.terminated, self.truncated, info
    
    def get_state(self):
        """
        Return the current state of the environment as a dictionary. Hashing of the state is done in the MCTS class.
        """
        # return {"rewards": self.unwrapped.current_rewards.copy(), "obstacles": self.unwrapped.obstacles.copy(), "agent_position": self.unwrapped.agent_position.copy()} 
        return {"rewards": self.current_rewards.copy(), "obstacles": self.obstacles.copy(), "agent_position": self.agent_position.copy()}
    
    def render(self, mode="human"):
        """
        Render the grid environment.
        """

        if mode == "human":
            visualize_rewards(self.current_rewards, self.obstacles, curr_pos=self.agent_position,start=self.start_pos,goal=self.goal_pos)

        else:
            grid = np.zeros_like(self.rewards, dtype=str)
            grid[self.obstacles == 1] = "X"  # Obstacles
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] == "":
                        grid[x, y] = f"{self.current_rewards[x, y]:.1f}" if self.current_rewards[x, y] > 0 else "."
            x, y = self.agent_position
            grid[x, y] = "A"  # Agent
            print("\n".join([" ".join(row) for row in grid]))
            print()

    def get_agent_position(self):
        return self.agent_position
    
    def get_current_rewards(self):
        return self.current_rewards
    
    def copy(self):
        return copy.deepcopy(self)
    
    def state_space_size(self):
        print('This is an upper bound on the size of the state space')

        # Get number of positive rewards
        num_reward_cells = np.sum(self.rewards > 0)

        # Get number of obstacles

        num_obstacles = np.sum(self.obstacles)

        total_states = (self.height*self.width - num_obstacles) * 2**num_reward_cells 

        return total_states

    


class WrapForMCTS(gym.ObservationWrapper):
    """
    Wrapper to make the Gridworld environment compatible with MCTS.
    Assigns a unique identifier (UID) to each observation for efficient hashing.
    """

    def __init__(self, env):
        super().__init__(env)

        self.state_to_uid = {}  # Dictionary to store unique identifiers for observations
        self.uid_counter = 0  # Simple integer UID counter

    def observation(self, observation):
        """
        Convert an observation into a UID for efficient lookups.
        """
        # obs_hash = self._compute_hash(observation)  # Generate a hash key for the state

        reward = tuple(observation["rewards"].flatten()) 
        obstacles = tuple(observation["obstacles"].flatten())
        agent_position = tuple(observation["agent_position"].flatten())

        #obs_hash = (reward,obstacles,agent_position)
        obs_hash = (obstacles,agent_position)


        return obs_hash

    def _compute_hash(self, observation):
        """
        Compute a stable hash from the observation using a hash function.
        """
        obs_str = str(self._make_hashable(observation)).encode("utf-8")  # Convert to bytes
        return hashlib.md5(obs_str).hexdigest()  # Generate a unique hash

    def _make_hashable(self, d):
        """Recursively convert observations into a hashable format."""
        return tuple(
            (k, self._convert_value(v)) for k, v in sorted(d.items())
        )

    def _convert_value(self, v):
        """Converts values to hashable types."""
        if isinstance(v, dict):
            return self._make_hashable(v)  # Recursively convert nested dicts
        elif isinstance(v, list) or isinstance(v, set):
            return tuple(v)  # Convert lists/sets to tuples
        elif isinstance(v, np.ndarray):
            return tuple(v.flatten())  # Convert NumPy arrays to tuples
        return v  # Scalars (int, float, str, bool) are already hashable
    
    def copy(self):
        return copy.deepcopy(self)
    

    def get_state(self):
        return self.unwrapped.get_state()
    
    def get_state_space_size(self):
        return self.unwrapped.state_space_size()


if __name__ == "__main__":
    import time
    import copy

    # test deepcopy time 
    test_times = []
    for i in range(1000): 
        rewards = np.ones((50,50))
        obstacles = np.ones((50,50))
        env = GridworldEnv(rewards, obstacles)
        env.reset()
        env.step(2)
        start = time.time()
        copy_env = env.copy()

        test_times.append(time.time()-start)

    print(f"Average time to copy environment: {np.mean(test_times)}")

    # test deepcopy time

    test_times = []
    for i in range(1000): 
        rewards = np.ones((50,50))
        obstacles = np.ones((50,50))
        env = GridworldEnv(rewards, obstacles)
        env.reset()
        env.step(2)
        start = time.time()
        copy_env = copy.deepcopy(env)
        test_times.append(time.time()-start)

    print(f"Average time to copy environment: {np.mean(test_times)}")








    

