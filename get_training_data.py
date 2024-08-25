import pickle
from utils import *

n = 10
config = "block"
num_blocks = 3
num_obstacles = 3
obstacle_type = "block"
square_size = 4

rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size)

with open('obstacle.pkl', 'wb') as f:
    pickle.dump(obstacles_map, f)
