import pickle
from utils import *
import numpy as np
import os 
from copy import deepcopy
from fo_solver import value_iteration,extract_policy
from nn_training import reformat_input,reformat_output
from tqdm import tqdm
import torch

total_init_dists = 100000 # number fo init distributions
seeds = np.arange(1,total_init_dists+1)

# np.random.shuffle(seeds)
# train_samples = seeds[:50000]
# val_samples = seeds[50000:]


def reformat_input_for_mpnet(obstacle,rewards):
    masked_reward_map = np.where(obstacle, -1, rewards)
    input = torch.tensor(masked_reward_map).unsqueeze(0).unsqueeze(0)  # Adding both channel and batch size
    
    assert input.shape == (1, 1, 10, 10)  # (batch_size, channels, height, width)

    return input

def get_action_map(agent_pos, next_pos):
    """Return int for diff in coords."""
    # Define action map: (dy, dx) -> action_index
    action_map = {
        (-1, 0): 0,  # up
        (0, 1): 1,   # right
        (1, 0): 2,   # down
        (0, -1): 3   # left
    }

    # Get the difference between the next position and the agent position
    action = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])

    return torch.tensor(agent_pos).unsqueeze(0), action_map[action]


def save_mpnet_style_data(id,input,coords,action,V,save_dir):
    torch.save((input,coords,action,V),os.path.join(save_dir,f"map_{id}.pt"))


def save_images(id,input,label,save_dir):
    torch.save((input, label), os.path.join(save_dir, f"map_{id}.pt"))



def cnn_training_data(obstacle_map,save_dir,seeds):
    
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

        
def auto_encoder_training_data(obstacle_map,save_dir,seeds):
    """Generate training data for enconder decoder.
    """

    id = 0
    for seed in tqdm(seeds,desc="Generate training data"):
        reward, obstacle_map = init_map(n,config,num_blocks,num_obstacles,obstacle_map=obstacle_map,seed=seed)
        start, goal = pick_start_and_goal(reward,obstacle_map,seed=seed)
        neighbors = precompute_next_states(n,obstacle_map)
        agent_pos = deepcopy(start)

        reward[agent_pos[0], agent_pos[1]] = 0

        V = value_iteration(n,rewards=reward, obstacles=obstacle_map,neighbors=neighbors,gamma=gamma)        

        policy = extract_policy(V,obstacle_map,neighbors,n=n)
        next_pos = tuple(int(i) for i in policy[agent_pos])


        input = reformat_input_for_mpnet(obstacle_map,reward)
        coords, action = get_action_map(agent_pos,next_pos)

        save_mpnet_style_data(id,input,coords, action,torch.tensor(V),save_dir)

        id+=1
        agent_pos = next_pos

        prev_reward = reward.copy()
        max_steps = 100
        steps = 0


        while np.any(reward > 0) and steps < max_steps:
            reward[agent_pos[0],agent_pos[1]] = 0
            V = value_iteration(n,rewards=reward, obstacles=obstacle_map,neighbors=neighbors,gamma=gamma)

            policy = extract_policy(V,obstacle_map,neighbors,n=n)
            next_pos = tuple(int(i) for i in policy[agent_pos])

            input = reformat_input_for_mpnet(obstacle_map,reward)
            coords, action = get_action_map(agent_pos,next_pos)
            save_mpnet_style_data(id,input,coords,action,torch.tensor(V),save_dir)

            id+=1
            agent_pos = next_pos
            steps+=1 




    



def main(obstacle_map,save_dir,seeds):
    os.makedirs(save_dir,exist_ok=True)

    print("Generating training data")
    #cnn_training_data(obstacle_map,save_dir=save_dir,seeds = seeds)
    auto_encoder_training_data(obstacle_map,save_dir,seeds)
    print("DONE")


if __name__ == "__main__":
    with open("obstacle.pkl","rb") as f:
        obstacle_map = pickle.load(f)


    save_dir = "training_data/auto_encoder_full_paths_data_with_values"
    main(obstacle_map,save_dir,seeds)
