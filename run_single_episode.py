from pytorch_value_iteration_networks.model import VIN
from utils import init_random_reachable_map,pick_start_and_goal
from copy import deepcopy
from vin_vs_vi import get_vin_path, get_vi_path, get_vin_replan_path
from nn_training import reformat_input
import torch
from fo_solver import visualize_rewards
from compare_paths import vin_replanner

if torch.cuda.is_available():
    device = torch.device('cuda')


else:
    device = torch.device('cpu')


def run_single_episode(model, heuristic=False,viz=False):
    actions = {0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)}  # up, right, down, left

    n = 20 # size of the grid
    min_obstacles = 2 # minimum number of obstacles
    max_obstacles = 20 # maximum number of obstacles

    max_steps = 500 # maximum number of steps to take
    step = 0 

    rewards,obstacles_map = init_random_reachable_map(n, 
                                "block", 
                                min_obstacles, 
                                max_obstacles, 
                                obstacle_type="block", 
                                obstacle_map=None, 
                                seed=None,
                                num_reward_blocks=(3,6),
                                reward_square_size=(2,8),
                                obstacle_cluster_prob=0.3,
                                obstacle_square_sizes=(1,5))
    
    start, goal = pick_start_and_goal(rewards, obstacles_map,seed=None)

    agent_pos = deepcopy(start)
    path = [agent_pos]

    ######### Heuristic: VIN + A* #########

    if heuristic:
        vin_path = vin_replanner(model, n, obstacles_map, rewards.copy(), start, goal, k=50)

        if viz:
            prev_pos = start
            for p in vin_path:
                rewards[p] = 0
                visualize_rewards(rewards,obstacles_map,start,goal,p,prev_pos)
                prev_pos = p
                if p == goal:
                    break
        return vin_pathb
    
    ######### Pure VIN #########

    while agent_pos != goal and step < max_steps:



        rewards[agent_pos[0], agent_pos[1]] = 0
        input = reformat_input(rewards, obstacles_map)
        input = input.unsqueeze(0)
        assert input.shape == (1, 2, n, n)

        logits, _, _ = model(
                        input.to(device).type(torch.float32),  # Ensure input is float32 and on the device
                        torch.tensor(agent_pos[0], dtype=torch.float32).to(device),  # Explicitly set dtype to float32
                        torch.tensor(agent_pos[1], dtype=torch.float32).to(device),  # Explicitly set dtype to float32
                        50)

        pred = torch.argmax(logits).item()
        action = actions[pred]

        new_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])

        if viz:
            visualize_rewards(rewards, obstacles_map, start, goal, agent_pos, new_pos)

        if obstacles_map[new_pos[0], new_pos[1]]:
            print("Agent moved into an obstacle")
            break

        agent_pos = new_pos
        path.append(agent_pos)
        step += 1

    return path


# def run_single_heuristic_episode(rewards, obstacles_map, start, goal, viz=False):
#      vin_path = vin_replanner(vin, n, obstacles_map, rewards.copy(), start, goal, k=50)



#     for p in vin_path:
#         rewards[p] = 0
#         visualize_rewards(rewards,obstacles_map,p,goal)
#         if p == goal:
#             break


def main(config,heuristic=True,viz=True):

    model = VIN(config)
    model.load_state_dict(torch.load(config.weights, map_location=device,weights_only=True))

    path = run_single_episode(model, heuristic, viz)


if __name__ == "__main__":  
    import argparse
    


    # vin_weights = torch.load('pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', map_location=device,weights_only=True)
    parser = argparse.ArgumentParser()

    ### Gridworld parameters

    parser.add_argument('--n', type=int, default=20, help='Grid size')
    parser.add_argument('--obstacle_shape', type=str, default="block", help='Shape of obstacles')
    parser.add_argument('--num_obstacles', type=int, default=5, help='Number of obstacles')
    parser.add_argument('--min_obstacles', type=int, default=2, help='Minimum obstacles')
    parser.add_argument('--max_obstacles', type=int, default=10, help='Maximum obstacles')
    parser.add_argument('--obstacle_type', type=str, default="block", help='Type of obstacles')
    parser.add_argument('--square_size', type=int, default=25, help='Size of the grid square')
    parser.add_argument('--obstacle_map', default=None, help='Initial obstacle map')
    #parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_reward_blocks', type=tuple, default=(2, 5), help='Range of reward blocks')
    parser.add_argument('--reward_square_size', type=tuple, default=(4, 6), help='Size of reward squares')
    parser.add_argument('--obstacle_cluster_prob', type=float, default=0.3, help='Probability of obstacle clustering')
    parser.add_argument('--obstacle_square_sizes', type=tuple, default=(3, 8), help='Range of obstacle square sizes')
    parser.add_argument('--living_reward', type=float, default=-0.1, help='Living reward for each step')
    parser.add_argument('--weights', default='pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', help='Path to weights file')

      
    # VIN-specific parameters
    parser.add_argument('--k', type=int, default=50, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_sz', type=int, default=1, help='Batch size')

    config = parser.parse_args()

    ######## Set heuristic to True to run the heuristic, False to run the pure VIN ########
    ######## Set viz to True to visualize the path ########
    main(config,heuristic=True,viz=True)