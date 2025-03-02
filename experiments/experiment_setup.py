import multiprocessing.pool
import time

import sys
import os
import pathlib
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

import multiprocessing
from multiprocessing import Pool

import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import init_random_reachable_map
from new_grid_env import GridworldEnv,WrapForMCTS
from fo_solver import visualize_rewards


def read_config(config_path):
    """Read config from YAML file and return as dictionary."""
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)
    return config

# def make_env(seed,config):



#     n = config["n"]
#     rewards_config = config["rewards_config"]
#     min_obstacles = config["min_obstacles"]
#     max_obstacles = config["max_obstacles"]

#     num_reward_blocks = config["num_reward_blocks"]
#     reward_square_size = config["reward_square_size"]
#     obstacle_type = config["obstacle_type"]

#     #obstacle_map = config["obstacle_map"]
#     obstacle_cluster_prob = config["obstacle_cluster_prob"] 
#     obstacle_square_sizes = config["obstacle_square_sizes"] 

#     living_reward = config["living_reward"]


#     rewards,obstacles_map = init_random_reachable_map(n=n,
#                                                       rewards_config=rewards_config,
#                                                       min_obstacles=min_obstacles,
#                                                       max_obstacles=max_obstacles,
#                                                       obstacle_type=obstacle_type,
#                                                       seed=seed,
#                                                       num_reward_blocks=num_reward_blocks,
#                                                       reward_square_size=reward_square_size,
#                                                       obstacle_cluster_prob=obstacle_cluster_prob,    
#                                                       obstacle_square_sizes=obstacle_square_sizes
#                                                       )
    
    
#     collision_penalty = config["collision_penalty"]
    
#     env = GridworldEnv(rewards,obstacles_map,start_pos=(0,0),goal_pos=(n-1,n-1),living_reward=living_reward,collision_penalty=collision_penalty)
#     wrapped_env = WrapForMCTS(env)
#     return wrapped_env

def make_env(seed,config):
    """This is simply a function to create a static environment for the 
    """
    
    n = config["n"]
    rewards_config = config["rewards_config"]
    min_obstacles = config["min_obstacles"]
    max_obstacles = config["max_obstacles"]

    num_reward_blocks = config["num_reward_blocks"]
    reward_square_size = config["reward_square_size"]
    obstacle_type = config["obstacle_type"]

    #obstacle_map = config["obstacle_map"]
    obstacle_cluster_prob = config["obstacle_cluster_prob"] 
    obstacle_square_sizes = config["obstacle_square_sizes"] 

    living_reward = config["living_reward"]


    if config.get("static_env",False): # If static env 
        seed = config["env_seed"]


    rewards,obstacles_map = init_random_reachable_map(n=n,
                                                      rewards_config=rewards_config,
                                                      min_obstacles=min_obstacles,
                                                      max_obstacles=max_obstacles,
                                                      obstacle_type=obstacle_type,
                                                      seed=seed,
                                                      num_reward_blocks=num_reward_blocks,
                                                      reward_square_size=reward_square_size,
                                                      obstacle_cluster_prob=obstacle_cluster_prob,    
                                                      obstacle_square_sizes=obstacle_square_sizes
                                                      )
    
    
    collision_penalty = config["collision_penalty"]


    if config.get("static_env",False): # If static env 
        rewards = np.zeros(shape=(n,n))

        reward_coords = config["reward_coords"]
        reward_values = config["reward_values"]
        
        for ind,(x,y) in enumerate(reward_coords):
            rewards[x,y] = reward_values[ind]

    env = GridworldEnv(rewards,obstacles_map,start_pos=(0,0),goal_pos=(n-1,n-1),living_reward=living_reward,collision_penalty=collision_penalty)
    wrapped_env = WrapForMCTS(env)
    return wrapped_env



def run_episode(config,new_agent_func,seed,max_steps):
    result = {
        "seed": seed,
        "reward": 0,
        "steps": 0,
        "time": 0,
        "collisions": 0,
        "found_all_rewards": 0,
        "max_steps": 0,
        "num_reward_blocks": 0,
        "num_obstacle_blocks": 0,
        "trajectory": []
    }
    
    
    # make env
    env = make_env(seed,config)
    obs,_ = env.reset(seed=seed)

    result["trajectory"].append(obs[1])

    state_dict = env.get_state()
    num_reward_blocks = np.sum(state_dict["rewards"] > 0)
    num_obstacle_blocks = np.sum(state_dict["obstacles"])
    result["num_reward_blocks"] = num_reward_blocks
    result["num_obstacle_blocks"] = num_obstacle_blocks

    # make agent
    agent = new_agent_func(env,obs,config,seed=seed)

    done = False
    total_reward = 0
    steps = 0
    collisions = 0


    start = time.time() 
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs,reward,done,_,info = env.step(action)
        current_position = obs[1]
        if hasattr(agent,"history"):
            agent.update_history(current_position)

        result["trajectory"].append(current_position)

        total_reward += reward
        steps += 1

        if info["collision"]:
            collisions += 1

    end = time.time()

    result["seed"] = seed
    result["reward"] = total_reward
    result["steps"] = steps
    result["time"] = end - start
    result["collisions"] = collisions

    if done:
        result["found_all_rewards"] = 1
    else:
        result["found_all_rewards"] = 0

    if steps == max_steps:
        result["max_steps"] = 1
    else:
        result["max_steps"] = 0

    result["reward"] = total_reward
    result["steps"] = steps
    result["time"] = end - start
    result["collisions"] = collisions
    result["found_all_rewards"] = int(done)
    result["max_steps"] = int(steps == max_steps)

    return result


def run_static_episode(config,new_agent_func,seed,max_steps):
    result = {
        "seed": seed,
        "reward": 0,
        "steps": 0,
        "time": 0,
        "collisions": 0,
        "found_all_rewards": 0,
        "max_steps": 0,
        "num_reward_blocks": 0,
        "num_obstacle_blocks": 0,
        "trajectory": []
    }
    
    
    # make env
    env = make_env(seed,config)
    obs,_ = env.reset(seed=seed)

    result["trajectory"].append(obs[1])

    state_dict = env.get_state()
    num_reward_blocks = np.sum(state_dict["rewards"] > 0)
    num_obstacle_blocks = np.sum(state_dict["obstacles"])
    result["num_reward_blocks"] = num_reward_blocks
    result["num_obstacle_blocks"] = num_obstacle_blocks

    # make agent
    agent = new_agent_func(env,obs,config,seed=seed)

    done = False
    total_reward = 0
    steps = 0
    collisions = 0

    start = time.time() 
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs,reward,done,_,info = env.step(action)
        current_position = obs[1]
        if hasattr(agent,"history"):
       
            agent.update_history(current_position)

        result["trajectory"].append(current_position)

        total_reward += reward
        steps += 1

        current_position = obs[1]


        if info["collision"]:
            collisions += 1

    end = time.time()

    result["seed"] = seed
    result["reward"] = total_reward
    result["steps"] = steps
    result["time"] = end - start
    result["collisions"] = collisions

    if done:
        result["found_all_rewards"] = 1
    else:
        result["found_all_rewards"] = 0

    if steps == max_steps:
        result["max_steps"] = 1
    else:
        result["max_steps"] = 0

    result["reward"] = total_reward
    result["steps"] = steps
    result["time"] = end - start
    result["collisions"] = collisions
    result["found_all_rewards"] = int(done)
    result["max_steps"] = int(steps == max_steps)

    return result


def run_episode_wrapper(args):
    return run_episode(*args)

def run_static_episode_wrapper(args):
    return run_static_episode(*args)

def save_file(results, filename):
    df = pd.DataFrame(results)  # Convert list of dictionaries to a DataFrame
    df.to_csv(filename, index=False)  # Save to CSV without row indices



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_file_to_json(results,filename):
    with open(filename,"w") as f:
        json.dump(results,f,indent=4,cls=NumpyEncoder)


def run_experiment(config,new_agent_func,static=False):

    # Experiment parameters 
    num_seeds = config["num_seeds"]
    seed_range = config["seed_range"]
    num_workers = config["num_workers"]
    max_steps = config["max_steps"]

    assert num_seeds <= (seed_range[1]-seed_range[0]), "Too many seeds bruh"

    seeds = [s for s in range(seed_range[0],seed_range[0]+num_seeds)]
    params = [(config,new_agent_func,seed,max_steps) for seed in seeds]

    results = []
    print("num_workers ", num_workers)
    
    if static:
        with Pool(num_workers) as pool, tqdm(total=len(params)) as pbar:
            for result in pool.imap_unordered(run_static_episode_wrapper, params):
                results.append(result)
                pbar.update(1)  

    else:
        with Pool(num_workers) as pool, tqdm(total=len(params)) as pbar:
            for result in pool.imap_unordered(run_episode_wrapper, params):
                results.append(result)
                pbar.update(1)  

    filepath = os.path.abspath(__file__)

    dir_name = os.path.dirname(filepath)
    new_dir = os.path.join(dir_name,"results")
    os.makedirs(new_dir, exist_ok=True)

    # filename = "experiments/results/" + config["experiment_name"] + ".csv"
    filename = "experiments/results/" + config["experiment_name"] + ".json"
    path = pathlib.Path(filename)
    save_file_to_json(results,path)  # Save results to CSV







