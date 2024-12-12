import numpy as np
import torch
import heapq
from collections import defaultdict



class ARAStarVIN:
    def __init__(self, vin_model, env, start, gamma=0.99, initial_weight=2.0, final_weight=1.0, weight_decay=0.9, max_steps=150, reward_threshold=10.0):
        self.vin_model = vin_model  # VIN model instance
        self.env = env  # Grid environment
        self.start = start  # Starting position
        self.gamma = gamma  # Discount factor for future rewards
        self.weight = initial_weight  # Initial weight on the heuristic
        self.final_weight = final_weight  # Final desired weight (optimal solution)
        self.weight_decay = weight_decay  # Decay rate for weight reduction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps  # Limit on the number of steps to explore
        self.reward_threshold = reward_threshold  # Target reward to achieve before stopping
        
        # Priority queue and other structures
        self.open_list = []
        self.closed_list = set()
        self.g_values = defaultdict(lambda: float('-inf'))  # Max reward collected to each state
        self.g_values[start] = 0  # Initial reward is 0 at the start
        self.vin_model.to(self.device)

    def heuristic(self, coords):
        """Use the VIN model's predicted values as a heuristic."""
        input_tensor, state_x, state_y = self.env.get_vin_input(coords)
        input_tensor = input_tensor.to(self.device)
        state_x = state_x.to(self.device)
        state_y = state_y.to(self.device)
        
        with torch.no_grad():
            _, _, value_map = self.vin_model(input_tensor, state_x, state_y, k=50)
        
        # Return the value of the current state 
        return value_map[0, 0, coords[0], coords[1]].cpu().item()

    def weighted_f(self, g, h):
        """Calculate the weighted f-value for ARA*."""
        return g + self.weight * h

    def search(self):
        """Run the ARA* search with iterative weight reduction."""
        # Initialize open list with the start state
        total_reward = 0
        steps = 0
        current = self.start
        start_h = self.heuristic(self.start)
        heapq.heappush(self.open_list, (self.weighted_f(0, start_h), self.start))
        
        # Continue exploring until conditions are met

        print("Starting search")
        while self.open_list and total_reward < self.reward_threshold and steps < self.max_steps:
            # If weight reaches final target, stop updating and commit to the solution found
            # if self.weight <= self.final_weight:
            #     break
            
            # Pop the node with the lowest f-value from the priority queue
            _, current = heapq.heappop(self.open_list)
            
            # Skip already-processed nodes
            if current in self.closed_list:
                continue

            self.closed_list.add(current)
            
            # Collect reward at the current state
            reward = self.env.get_reward(current)  

            total_reward += reward
            steps += 1

            if steps % 10 == 0:
                print(f"Step {steps}: Total reward collected = {total_reward}")

            # Expand neighbors to find more reward-rich areas
            valid_actions = self.env.get_valid_actions(current)
            for action in valid_actions:
                next_state = self.env.get_next_state(current, action)
                
                # Calculate the cumulative reward for the neighbor
                immediate_reward = self.env.get_reward(next_state)
                g_value = self.g_values[current] + immediate_reward
                h_value = self.heuristic(next_state)
                
                # Update the path to the neighbor if a higher g-value is found
                if g_value > self.g_values[next_state]:
                    self.g_values[next_state] = g_value
                    f_value = self.weighted_f(g_value, h_value)
                    heapq.heappush(self.open_list, (f_value, next_state))
            
            # Reduce weight iteratively after each expansion
            self.weight *= self.weight_decay
        
        # Return the path taken and the total reward collected

        print(f"Search complete. Total reward collected = {total_reward}")
        print("Reconstructing path...")
        return self.reconstruct_path(current), total_reward

    def reconstruct_path(self, end_state):
        """Reconstruct the path to the final state reached within the step or reward limits."""
        path = [end_state]
        current = end_state

        print("end state", end_state)

        i = 0

        # Trace back from the end state to the start
        while current != self.start:
            current = max(
                (self.env.get_next_state(current, action) for action in self.env.get_valid_actions(current)),
                key=lambda s: self.g_values[s]
            )
            print(f"added_state {i}")
            i += 1
            path.append(current)
        path.reverse()
        return path



if __name__ == "__main__":


    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    from fo_solver import init_random_reachable_map,pick_start_and_goal
    from pytorch_value_iteration_networks.model import VIN
    from modified_gridenv import ModifiedGridEnvironment
    import argparse

    successful_paths = []
    failed_paths = []
    n = 20
    seeds = np.random.randint(6100, 20000, 100)
    # seeds = [9651]

    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True, map_location=device)
 
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

      
    # VIN-specific parameters
    parser.add_argument('--k', type=int, default=50, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_sz', type=int, default=1, help='Batch size')

    config = parser.parse_args()


    vin = VIN(config)
    vin.load_state_dict(vin_weights)

    vin.to(device)
    vin.eval()
    
    rewards, obstacles_map = init_random_reachable_map(n, "block",3, 2, 20, obstacle_type="block", square_size=4, obstacle_map=None)
    start ,target = pick_start_and_goal(rewards, obstacles_map)

    env = ModifiedGridEnvironment(config,rewards,obstacles_map,train=False, start=start, target=target, living_reward=None, shuffle=False, max_steps=100)
    agent = ARAStarVIN(vin, env, start=(0, 0),reward_threshold=sum(rewards.flatten()),max_steps=150)

    path, total_reward = agent.search()






