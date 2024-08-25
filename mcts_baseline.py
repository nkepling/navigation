from ns_gym.benchmark_algorithms import MCTS
from q_learning import GridEnvironment
from utils import *
from fo_solver import visualize_rewards

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
