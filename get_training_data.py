import pickle
from utils import *
import numpy as np
import h5py
import os 
import sqlite3
from copy import deepcopy
from fo_solver import value_iteration,extract_policy
from nn_training import reformat_input,reformat_output
from tqdm import tqdm

total_init_dists = 60000
seeds = np.arange(1,total_init_dists+1)

# np.random.shuffle(seeds)
# train_samples = seeds[:50000]
# val_samples = seeds[50000:]




def save_images(id,input,label,save_dir):
    torch.save((input, label), os.path.join(save_dir, f"map_{id}.pt"))



def cnn_training_data(obstacle_map,save_dir):
    
    id = 0
    

    for seed in tqdm(seeds,desc="Generate training data"):
        reward, obstacle_map = init_map(n,config,num_blocks,num_obstacles,obstacle_map=obstacle_map,seed=seed)
        start, goal = pick_start_and_goal(reward,obstacle_map,seed=seed)
        neighbors = precompute_next_states(n,obstacle_map)
        agent_pos = deepcopy(start)

        reward[agent_pos[0], agent_pos[1]] = 0

        V = value_iteration(n,rewards=reward, obstacles=obstacle_map,neighbors=neighbors,gamma=gamma)

        input = reformat_input(reward,obstacle_map)
        label = reformat_output(V)
        save_images(id,input,label,save_dir)
        id+=1

        policy = extract_policy(V,obstacle_map,neighbors,n=n)
        next_pos = tuple(int(i) for i in policy[agent_pos])
        agent_pos = next_pos

        prev_reward = reward.copy()
        max_steps = 100
        steps = 0


        while agent_pos != goal and steps < max_steps:
            reward[agent_pos[0],agent_pos[1]] = 0
            V = value_iteration(n,rewards=reward, obstacles=obstacle_map,neighbors=neighbors,gamma=gamma)

            if prev_reward[agent_pos[0],agent_pos[1]] !=0:
                input = reformat_input(reward,obstacle_map)
                label = reformat_output(V)
                save_images(id,input,label,save_dir)
                id+=1
                prev_reward = reward.copy()

            policy = extract_policy(V,obstacle_map,neighbors,n=n)
            next_pos = tuple(int(i) for i in policy[agent_pos])
            agent_pos = next_pos
            steps+=1 

        


def main(obstacle_map):
    os.makedirs("training_data_with_reward_updates",exist_ok=True)
    save_dir = f"training_data_with_reward_updates"

    print("Generating training data for VIN model")
    cnn_training_data(obstacle_map,save_dir=save_dir)
    print("DONE")


if __name__ == "__main__":
    with open("obstacle.pkl","rb") as f:
        obstacle_map = pickle.load(f)
    main(obstacle_map)
