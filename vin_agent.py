from pytorch_value_iteration_networks.model import VIN
import torch
import numpy as np
from types import SimpleNamespace

class VINAgent:
    def __init__(self, vin_model_path, config, device='cpu'):
        self.device = torch.device(device)

        if not isinstance(config,SimpleNamespace):
            config = SimpleNamespace(**config)

        self.vin = VIN(config)
        self.vin.load_state_dict(torch.load(vin_model_path, map_location=self.device))
        self.vin.to(self.device)
        self.vin.eval()
        self.config = config

        self.action_to_dir = {0: np.array([0, -1]), 1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([-1, 0])}

    def reformat_input(self, rewards, obstacles):
        """Reformat the input for the VIN model"""
        temp = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
        obstacles_map = torch.tensor(obstacles, dtype=torch.float32).unsqueeze(0)
        input = torch.cat((temp, obstacles_map), dim=0)
        input = input.unsqueeze(0)
        n = len(obstacles)
        assert input.shape == (1, 2, n, n)
        return input

    def vin_rollout(self, env, depth=10):
        """Use the trajectory from the VIN to do a rollout rather than the value estimate"""
        vin_trajectory = []
        for _ in range(depth):
            state_dict = env.unwrapped.get_state()
            input = self.reformat_input(state_dict["rewards"], state_dict["obstacles"])
            logits, probs, value = self.vin(input, torch.tensor(state_dict["agent_position"][0]), torch.tensor(state_dict["agent_position"][1]), k=self.config.k)
            x = state_dict["agent_position"][0]
            y = state_dict["agent_position"][1]

            probs = probs.detach().numpy().squeeze()
            action = np.random.choice(len(probs), p=probs)
            observation, reward, terminated, truncated, info = env.step(action)

            vin_trajectory.append(reward * (self.config.gamma ** _))

            if terminated or truncated:
                break

        return sum(vin_trajectory)

    def act(self, env, depth=10):
        """Select an action using the VIN model"""
        state_dict = env.unwrapped.get_state()
        input = self.reformat_input(state_dict["rewards"], state_dict["obstacles"])
        logits, probs, value = self.vin(input, torch.tensor(state_dict["agent_position"][0]), torch.tensor(state_dict["agent_position"][1]), k=self.config.k)
        x = state_dict["agent_position"][0]
        y = state_dict["agent_position"][1]

        probs = probs.detach().numpy().squeeze()
        print("probs ", probs)
        print(value)
        action = np.random.choice(len(probs), p=probs)

        # action = np.argmax(probs)
        return action
        

    def get_best_action_from_value(self, env):
        """Get the best action based on the value estimates from the VIN model"""
        state_dict = env.unwrapped.get_state()
        input = self.reformat_input(state_dict["rewards"], state_dict["obstacles"])
        logits, probs, value = self.vin(input, torch.tensor(state_dict["agent_position"][0]), torch.tensor(state_dict["agent_position"][1]), k=self.config.k)
        x = state_dict["agent_position"][0]
        y = state_dict["agent_position"][1]

        agent_position = state_dict["agent_position"]

        value = value.detach().numpy().squeeze()

        next_state_values = []
        for a,dir in self.action_to_dir.items():
            neighbor = agent_position + dir
            if 0 <= neighbor[0] < value.shape[0] and 0 <= neighbor[1] < value.shape[1]:
                next_state_values.append(value[neighbor[0], neighbor[1]])
            else:
                next_state_values.append(value[agent_position[0],agent_position[1]])

        best_action = np.argmax(next_state_values)
        return best_action

if __name__ == "__main__":
    from experiments.experiment_setup import read_config, make_env
    from new_grid_env import GridworldEnv, WrapForMCTS
    from utils import init_random_reachable_map, pick_start_and_goal
    import argparse
    from fo_solver import visualize_rewards

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
    parser.add_argument('--vin_model_path', type=str, required=True, help='Path to the VIN model file')
    args = parser.parse_args()

    config = read_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vin_agent = VINAgent(args.vin_model_path, config, device=device)

    n = config["n"]
    min_obstacles = config["min_obstacles"]
    max_obstacles = config["max_obstacles"]
    config["env_seed"] = 27
    config["static"] = True

    env_seed = config["env_seed"]

    

    env = make_env(env_seed,config)



    total_reward = 0
    observation, _ = env.reset()
    max_steps = config["max_steps"]
    step = 0
    done = False

    while step < max_steps and not done:
        visualize_rewards(env.unwrapped.current_rewards,env.unwrapped.obstacles,env.unwrapped.agent_position,(4,4))
        action = vin_agent.act(env)
        # action = vin_agent.get_best_action_from_value(env)
        observation, reward, done, _, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"\rStep count {step}", end="", flush=True)

    print("Total reward: ", total_reward)