from fo_solver import * 
from utils import *


with open('obstacle.pkl', 'rb') as f:
    obstacles = pickle.load(f)

rewards,obstacles_map = init_map(n,config,num_blocks,num_obstacles,obstacle_type,square_size,obstacle_map=obstacles,seed=31415)

neighbors = precompute_next_states(n,obstacles_map)

start,goal = pick_start_and_goal(rewards,obstacles_map)
goal = (8,8)

visualize_rewards(rewards,obstacles_map,start,goal)




