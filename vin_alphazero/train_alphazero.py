import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch import optim
import torch.nn.functional as F

from vin_alphazero import VINAlphaZeroAgent
from gridworld_env import GridEnvironment
from utils import init_random_reachable_map, pick_start_and_goal
import argparse



def train_alphazero(config):
    """Train the AlphaZero agent using the Value Iteration Network and MCTS.
    """
    agent = VINAlphaZeroAgent(config)

    # for now justa 

    rewards, obstacles_map = init_random_reachable_map(
                                                    config.n,
                                                    config.obstacle_shape,
                                                    config.num_obstacles,
                                                    config.min_obstacles,
                                                    config.max_obstacles,
                                                    obstacle_type=config.obstacle_type,
                                                    square_size=config.square_size,
                                                    obstacle_map=config.obstacle_map,
                                                    seed=config.seed,
                                                    num_reward_blocks=config.num_reward_blocks,
                                                    reward_square_size=config.reward_square_size,
                                                    obstacle_cluster_prob=config.obstacle_cluster_prob,
                                                    obstacle_square_sizes=config.obstacle_square_sizes)
    
    start,target = pick_start_and_goal(rewards,obstacles_map)

    env = GridEnvironment(config, rewards, obstacles_map, start, target, config.living_reward, shuffle=False, train=True)

    agent.train(config,env)    



def main(config):
    train_alphazero(config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Size of replay buffer')

    
    # VIN-specific parameters
    parser.add_argument('--k', type=int, default=50, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')

    # MCTS parameters
    parser.add_argument('--max_mcts_search_depth', type=int, default=10, help='Maximum depth of MCTS search')
    parser.add_argument('--num_mcts_simulations', type=int, default=100, help='Number of MCTS simulations per step')
    parser.add_argument('--c', type=float, default=1.0, help='Exploration constant in MCTS')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet noise parameter for exploration')
    parser.add_argument('--epsilon', type=float, default=0.25, help='Probability for Dirichlet noise addition')


    # Temperature parameters for AlphaZero-like training
    parser.add_argument('--temp_start', type=float, default=2.0, help='Initial temperature for exploration')
    parser.add_argument('--temp_end', type=float, default=0.8, help='Final temperature after decay')
    parser.add_argument('--temp_decay', type=float, default=0.95, help='Decay factor for temperature')

    # Evaluation parameters
    parser.add_argument('--eval_window_size', type=int, default=100, help='Window size for evaluating moving average of rewards')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--max_episode_len', type=int, default=100, help='Maximum length of each episode')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default='experiment_alpha_zero', help='Name for experiment logging')

    # Model checkpoint path
    parser.add_argument('--model_checkpoint_path', type=str, default=None, help='Path to model checkpoint for loading')

    # Environment parameters
    parser.add_argument('--n', type=int, default=20, help='Grid size')
    parser.add_argument('--obstacle_shape', type=str, default="block", help='Shape of obstacles')
    parser.add_argument('--num_obstacles', type=int, default=5, help='Number of obstacles')
    parser.add_argument('--min_obstacles', type=int, default=2, help='Minimum obstacles')
    parser.add_argument('--max_obstacles', type=int, default=10, help='Maximum obstacles')
    parser.add_argument('--obstacle_type', type=str, default="block", help='Type of obstacles')
    parser.add_argument('--square_size', type=int, default=25, help='Size of the grid square')
    parser.add_argument('--obstacle_map', default=None, help='Initial obstacle map')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_reward_blocks', type=tuple, default=(2, 5), help='Range of reward blocks')
    parser.add_argument('--reward_square_size', type=tuple, default=(4, 6), help='Size of reward squares')
    parser.add_argument('--obstacle_cluster_prob', type=float, default=0.3, help='Probability of obstacle clustering')
    parser.add_argument('--obstacle_square_sizes', type=tuple, default=(3, 8), help='Range of obstacle square sizes')
    parser.add_argument('--living_reward', type=float, default=-0.1, help='Living reward for each step')

    config = parser.parse_args()
    print(config.n)
    main(config)

    

