from ddqn import DQNAgent, CNN
from gridworld_env import GridEnvironment
from utils import * 
from fo_solver import visualize_rewards
from eval import visually_compare_value_functions, visually_compare_policy
import pickle
import torch
from collections import deque
import os
import numpy as np
from tqdm import tqdm
from utils import format_input_for_ddqn_cnn

n, config, num_blocks, num_obstacles, obstacle_type, square_size, random_map, gamma = parse_arguments()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def dim_check(state):
    if (state.shape == (3,10,10)):
        state = state.unsqueeze(0)
    assert(state.shape == (1,3,10,10))
    return state

def train(model_name, num_episodes, eps, eps_end, eps_decay, lr, max_steps_per_episode):

    with open('obstacle.pkl', 'rb') as f:
        obstacles_map = pickle.load(f)
    
    model = CNN().to(device)
    agent = DQNAgent(model=model, model_path=None, lr=lr)
    
    scores_window = deque(maxlen=100)
    best_score = -np.inf
    
    # Use tqdm for progress bar
    for ep in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
        rewards, obstacles_map = init_map(n, config, num_blocks, num_obstacles, obstacle_type, square_size, obstacle_map=obstacles_map)
        start, goal = pick_start_and_goal(rewards, obstacles_map)
        env = GridEnvironment(n, rewards, obstacles_map, start, goal)
        out = env.reset()
        state = env.get_cnn_input()  # Initialize the state
        score = 0
        t = 0
        
        while True:
            action, values = agent.act(format_input_for_ddqn_cnn(state).to(device), eps)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = env.get_cnn_input()
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done or truncated or t >= max_steps_per_episode:
                break
            t += 1
        
        scores_window.append(score)
        eps = max(eps_end, eps_decay * eps)

        if ep % 100 == 0:
            print(f"\nFinished episode {ep}")
            torch.save(agent.q_network_local.state_dict(), os.path.join(saved_model_dir, f"{model_name}_ep_{ep}"))

        if np.mean(scores_window) > best_score:
            best_score = np.mean(scores_window)
            print(f"\nBest model found at episode {ep} with score {best_score}")
            torch.save(agent.q_network_local.state_dict(), os.path.join(saved_model_dir, f"{model_name}_best"))

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(current_dir, 'DDQN_weights'), exist_ok=True)
    saved_model_dir = os.path.join(current_dir, 'DDQN_weights')
    model_name = "DQQN_Model"
    train(model_name, num_episodes=1500, eps=1.0, eps_end=0.01, eps_decay=0.997, lr=0.001, max_steps_per_episode=200)
