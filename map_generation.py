import numpy as np
from utils import *
from types import SimpleNamespace   


"""
This file is to generate the map for the drone to navigate that more closely resembles the AirSim environment.
"""

def create_obstacles_map(n,min_obstacles,max_obstacles,min_obstacles_size,max_obstacles_size,num_no_fly_zones,no_fly_zone_size,seed=None):
    

    if seed is not None:
        np.random.seed(seed)

    # Create empty map
    obstacles_map = np.zeros((n,n))


    # Create obstacles
    num_obstacles = np.random.randint(min_obstacles,max_obstacles)

    for _ in range(num_obstacles):
        obstacle_square_size = np.random.randint(min_obstacles_size,max_obstacles_size)
        start_x = random.randrange(0, n)
        start_y = random.randrange(0, n)
        end_x = min(start_x + obstacle_square_size, n)
        end_y = min(start_y + obstacle_square_size, n)
        obstacles_map[start_x:end_x, start_y:end_y] = True


    # get number of no fly zones

    num_no_fly_zones = np.random.randint(num_no_fly_zones)

    for _ in range(num_no_fly_zones):
        no_fly_zone_square_size = np.random.randint(no_fly_zone_size)
        start_x = random.randrange(0, n)
        start_y = random.randrange(0, n)
        end_x = min(start_x + no_fly_zone_square_size, n)
        end_y = min(start_y + no_fly_zone_square_size, n)
        obstacles_map[start_x:end_x, start_y:end_y] = True 

    return obstacles_map

    

def create_reward_map(num_areas_of_interest, max_area_of_interest_size, min_area_of_interest_size, obstacles_map,seed=None):
    if seed is not None:
        np.random.seed(seed)

    rewards_map = np.zeros(obstacles_map.shape)

    num_reward_blocks = np.random.randint(num_areas_of_interest)

    for _ in range(num_reward_blocks):
        area_of_interest_size = np.random.randint(min_area_of_interest_size,max_area_of_interest_size)
        start_x = random.randrange(0, n)
        start_y = random.randrange(0, n)
        end_x = min(start_x + area_of_interest_size, n)
        end_y = min(start_y + area_of_interest_size, n)
        rewards_map[start_x:end_x, start_y:end_y] = random.randint(1,10)


    # make sure the rewards add up to 1
    rewards_map = rewards_map / np.sum(rewards_map)
    return rewards_map



def generate_map(config):

    obstacles_map = create_obstacles_map(config.n,config.min_obstacles,config.max_obstacles,config.min_obstacles_size,config.max_obstacles_size,config.num_no_fly_zones,config.no_fly_zone_size,seed=None)
    rewards_map = create_reward_map(config.num_areas_of_interest, config.max_area_of_interest_size, config.min_area_of_interest_size, obstacles_map,seed=None)

    return obstacles_map, rewards_map



if __name__ == "__main__":

    config = SimpleNamespace()
    config.n = 400
    config.drone_height = 50
    config.num_areas_of_interest = (1, 5)

    config.min_obstacles = 10 






