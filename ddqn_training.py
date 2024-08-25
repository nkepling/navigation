from ns_gym.benchmark_algorithms import DDQN
from q_learning import GridEnvironment
from utils import * 
from fo_solver import visualize_rewards
from eval import visually_compare_value_functions, visually_compare_policy
import pickle
import torch




n,config,num_blocks,num_obstacles,obstacle_type,square_size,random_map,gamma= parse_arguments()

def train(num_episodes, env, agent, epsilon, gamma, lr, batch_size, target_update, replay_buffer_size, seed):

    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)

    rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size, obstacle_map=obstacles_map)
    neighbors = precompute_next_states(n, obstacles_map)
    agent = DDQN.DQNAgent()
    for ep in num_episodes:
        start,goal = pick_start_and_goal(rewards, obstacles_map)
        env = GridEnvironment(n, rewards, obstacles_map, start, goal)

        for step in range(max_steps_per_episode):
            position, visited,{} = env.reset()
            state = agent.get_state_index(position, visited)
            total_reward = 0

            for step in range(max_steps_per_episode):
                valid_actions = env.get_valid_actions()
                action = agent.choose_action(state, valid_actions)
                (next_position, next_visited), reward, done,_,_ = env.step(action)
                next_state = agent.get_state_index(next_position, next_visited)

                agent.update(state, action, reward, next_state, valid_actions)

                position, visited = next_position, next_visited
                state = next_state
                total_reward += reward

                if done:
                    break







