import torch 
import torch.nn as nn

import numpy as np
import time
import pickle
import os
import argparse

from pytorch_value_iteration_networks.model import *
from utils import *
from fo_solver import *

from gridworld_env import GridEnvironment
from nn_training import reformat_input, reformat_output

from collections import defaultdict,deque
import hashlib

from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def add(self, reward_map, S1, S2, value, pi):
        # Pack the experience with all required elements
        experience = (reward_map, S1, S2, value, pi)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
            self.idx = (self.idx + 1) % self.max_size

    def __len__(self):
        # Return the current number of items in the buffer
        return len(self.buffer)

    def __getitem__(self, idx):
        # Retrieve experience at index idx
        reward_map, S1, S2, value, pi = self.buffer[idx]

        # Convert to torch tensors
        reward_map = torch.tensor(reward_map, dtype=torch.float32)
        S1 = torch.tensor(S1, dtype=torch.long)
        S2 = torch.tensor(S2, dtype=torch.long)
        value = torch.tensor(value, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)

        return reward_map, S1, S2, value, pi



def softmax(logits):
    """Compute softmax values for the given logits."""
    max_logit = np.max(logits)  # For numerical stability
    exp_logits = np.exp(logits - max_logit)  # Subtract max logit to prevent overflow
    return exp_logits / np.sum(exp_logits)

def stable_normalizer(x,temp):
    x  = (x/(np.max(x)))**(1/temp)
    return np.abs(x/(np.sum(x)))

class UCTNode:
    def __init__(self, reward, coords = (0,0), prior_prob=0, parent=None,action=None,depth=0,val_init=0):
        self.visits = 0  # N(s,a) in PUCT
        self.init_value = 0 
        self.value_sum = 0  # W(s,a) - cumulative value
        self.prior_prob = prior_prob  # P(s,a) - prior probability from VIN logits
        self.parent = parent
        self.children = {}  # dictionary of actions to UCTNodes
        self.coords = coords  # coordinates of the state in the grid
        self.depth = 0 # depth of the node in the tree
        self.action = None # action taken to reach this node

        self.reward = reward # instantaneous reward
        self.val_init =  val_init # prediced value of the state

    def update(self, value):
        """Update the node with new value."""
        self.visits += 1
        self.value_sum += value

    def get_value(self, c_puct):
        """Calculate the PUCT value for this node."""

        # if self.visits == 0:
        #     return np.inf
        # else:
        u_value = c_puct * self.prior_prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        q_value = self.value_sum / (1 + self.visits)  # Q(s,a) = W(s,a) / N(s,a)
        return q_value + u_value

    def get_children(self):
        """Return all child nodes."""
        return self.children
    
    def expand(self, action_probs, valid_next_states, next_state_value, next_state_rewards):
        """Expand the node by creating child nodes for each possible action."""
        
        # Filter action_probs to include only valid actions
        valid_action_probs = {action: prob for action, prob in action_probs.items() if action in valid_next_states}
        
        # Renormalize the probabilities
        total_prob = sum(valid_action_probs.values())
        if total_prob > 0:
            valid_action_probs = {action: prob / total_prob for action, prob in valid_action_probs.items()}
        else:
            raise ValueError("Total probability of valid actions is zero, cannot renormalize.")

        for action, prob in valid_action_probs.items():
            child_coords = valid_next_states[action]
            assert isinstance(child_coords, tuple), f"child_coords is {child_coords}"
            reward = next_state_rewards[action]  # Immediate reward for the next state
            self.children[action] = UCTNode(
                reward=reward,
                coords=child_coords,
                prior_prob=prob,
                parent=self,
                action=action,
                depth=self.depth + 1,
                val_init=next_state_value[action],
            )

    # def expand(self, action_probs, valid_next_states,next_state_value,next_state_rewards):
    #     """Expand the node by creating child nodes for each possible action."""

    #     for action, prob in action_probs.items():
    #         if action in valid_next_states:
    #             child_coords = valid_next_states[action]
    #             assert isinstance(child_coords, tuple), f"child_coords is {child_coords}"
    #             reward = next_state_rewards[action] # next state immediate reward
    #             self.children[action] = UCTNode(reward,coords=child_coords, prior_prob=prob, parent=self,action=action,depth=self.depth+1,val_init=next_state_value[action])


    def is_leaf(self):
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0
    
class PUCT:
    def __init__(self, vin_model, cnn_model,env:GridEnvironment, start, gamma, c_puct=1.0, k=50,alpha=0.15,epsilon=0.25):
        self.vin_model = vin_model  # Value Iteration Network to get values and logits
        self.cnn_model = cnn_model  
        self.env = env  # The grid environment instance
        self.c_puct = c_puct
        self.k = k 
   
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        vin_model.to(self.device)
        cnn_model.to(self.device)
   
        # add dirichlet noise to the prior probabilities to encourage exploration

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        ## Keep track of previous action to add momentum based penalty to the PUCT value. 
        self.prev_action = None
        self.turn_penalty = 0.1

        self.Qsa = defaultdict(float)  # stores Q values for s,a
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited
        self.Ns = defaultdict(int)  # sto


    
    def search(self,root_coords,init_reward,num_simulations):
        """Run multiple simulations to build the tree.
        satrt: the start state of the agent (coords)
        """


        logits,preds,value = self._evaluate(UCTNode(reward=0,coords=root_coords))
        val_init = value[0,0,root_coords[0],root_coords[1]]
        self.root = UCTNode(reward=0,coords=root_coords,val_init=val_init)  # Root node of the tree

        for _ in range(num_simulations):
            # self.c_puct = max(0.5, self.c_puct * 0.99) 
            path_memory = defaultdict(int)

            
            node = self.root
            self.env.reset(node.coords,init_reward)
            # Traverse the tree until we reach a leaf node
            while not node.is_leaf():
                state_hash = self.env.encode_state(node.coords)
                path_memory[state_hash] += 1
                node = self._select(node, path_memory)

            # Expand and evaluate the leaf node
            value = self._expand_and_evaluate(node)

            # Backpropagate the value through the tree
            self._backpropagate(node)

        # best_action = self._get_best_action(self.root)

        best_action = max(self.root.get_children(), key=lambda x: self.root.get_children()[x].visits)

        # best_action = self._get_best_action(self.root,path_memory)

        best_child = self.root.get_children()[best_action]

        # best_child = max(self.root.get_children().values(), key=lambda x: x.get_value(0))   # get the child with the highest value

        # best_action = best_child.action 

        children = self.root.get_children()

        for c in children:
            print(f"Action: {c}, Visits: {children[c].visits}, Prior: {children[c].prior_prob}, Value: {children[c].get_value(0)}")

        print('################################')
        return best_action, best_child.coords
    
    def _get_best_action(self,node,path_memory):
        """Select the best action based on the visit counts."""
        best_puct_value = -np.inf
        best_action = None
        best_child = None
 
        for action, child in node.get_children().items():
            puct_value = child.get_value(self.c_puct)


        
            # ############ Add a penalty for to visited nodes to encourage exploration #########
            # penalty = 0.1 * child.visits
            # puct_value -= penalty


            # revisit_penalty = path_memory[self.encode_state(child.coords)] * 0.5
            # puct_value -= revisit_penalty


            ##### Turn penalty ########
            # if self.prev_action is not None and action != self.prev_action:
            #     puct_value -= self.turn_penalty

            if puct_value > best_puct_value:
                    best_puct_value = puct_value
                    best_action = action
                    best_child = child

        return best_action

    def _select(self, node,path_memory):
        """Select the child node with the highest PUCT value."""
        best_action = self._get_best_action(node,path_memory)
        self.prev_action = best_action
        self.env.step(best_action)
        return node.get_children()[best_action]
    
    def _evaluate(self, node):
        input,state_x,state_y = self.env.get_vin_input(node.coords)

        input = input.to(self.device)
        state_x = state_x.to(self.device)
        state_y = state_y.to(self.device)


        assert input.shape == (1,2,self.env.n,self.env.n), f"input shape is {input.shape}"
        logits, preds, _ = self.vin_model(input,state_x,state_y,self.k)

        # convert from torch tensor to numpy array
        logits = logits.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        #value = value.cpu().detach().numpy() # this is a value map ...


        value = self.cnn_model(input)

        return logits,preds,value
        

    def _expand_and_evaluate(self, node:UCTNode):
        """Expand the node using the VIN model and return the value."""
        # Run the VIN model to get values and logits for the current state
        logits,preds,value = self._evaluate(node)


        # Expand the node with the action probabilities
        valid_actions = self.env.get_valid_actions(node.coords)

        next_state_value = self._get_next_state_value(node,value,valid_actions) # dict of next state values for each action
    
        action_probs = preds.flatten()

        ############## Add dirichlet noise to the prior probabilities to encourage exploration if the node is the root node #########

        if node == self.root:
            print("Adding dirichlet noise to the prior probabilities")
            action_probs = (1-self.epsilon) * action_probs + self.epsilon * np.random.dirichlet([self.alpha] * len(action_probs))
        
        action_prob_dict = {i: action_probs[i] for i in range(len(action_probs))}
        reward_map = self.env.get_rewards()
        next_states_rewads = {a: reward_map[valid_next_state] for a, valid_next_state in self.env.get_valid_next_states(node.coords).items()}
        node.expand(action_prob_dict, self.env.get_valid_next_states(node.coords),next_state_value,next_states_rewads)

        return next_state_value

    # def _backpropagate(self, node):
    #     """Backpropagate the accumulated ground truth rewards up to the root."""

    #     # Initialize with the node's current ground truth reward
    #     r = node.val_init

    #     while node is not None:
    #         # Update the node with the current reward
    #         node.update(r)

    #         # Move to the parent node and apply the discount factor
    #         node = node.parent
    #         if node is not None:
    #             r = node.reward + self.gamma * r  # Accumulate the reward with discounting

    def _backpropagate(self,node):
        """Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """

        R  = node.val_init
        depth = 0
        while node.parent:
            # R = node.reward + self.gamma * R
            R = self.gamma**depth * R 
            node = node.parent
            depth += 1
            node.update(R)
        
            

    def _get_next_state_value(self, node, value, valid_actions):
        """Given value map grab next state value for current agent position

        Returns:
            next_state_value (dict): value of the next state for each each action
        """
        action_state_dict = self.env.get_valid_next_states(node.coords)

        values = np.zeros(4)

        for a in range(4):
            
            if a not in valid_actions:
                # values[a] = -np.inf
                values[a] = 0 #SET TO ZERO FOR NOW
            else:
                values[a] = value[0,0,action_state_dict[a][0],action_state_dict[a][1]]
        
        next_state_value = {a: values[a] for a in range(4)}

        return next_state_value
    
    def update_root(self, best_action):
        """Update the root node to the child node corresponding to the best action."""
        if best_action in self.root.get_children():
            # Set the new root to the child node corresponding to the best action
            new_root = self.root.get_children()[best_action]
            new_root.parent = None  # Remove the parent reference to make it the root
            self.root = new_root

    def encode_state(self, coords):
        """Encode the state as a hashable key."""
        hasher = hashlib.md5()
        hasher.update(str(coords).encode('utf-8'))
        return hasher.hexdigest()
    
    def return_results(self,temp):
        """Return the results of the MCTS search after it is finished.
        Args:
            temp (float): Parameter for getting the pi values.
        Returns:
            state (np.ndarray): Game state, usually as a 1D np.ndarray.
            pi (np.ndarray): Action probabilities from this state to the next state as collected by MCTS.
            V_target (float): Actual value of the next state.
        """
        
        #NOTE: THese have to be dicitionaries becuase we can have less than 4 actions... easier to keep track of the values..
        counts = np.array([child.visits for child in self.root.children.values()])  
        Q = np.array([child.value_sum/child.visits for child in self.root.children.values()])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q) #calcualte expected value of next state
        best_action = np.argmax(counts)
 
        return pi_target, V_target, best_action
    

"""
Hybid PUCT trainig -- alphazero style


Hybrid because we are starting with a pretrained model and then training it with the PUCT algorithm.. 
"""    

def update_models(vin_model,cnn_model,buffer,config):
    data_loader = torch.utils.data.DataLoader(buffer, batch_size=config.batch_sz, shuffle=True)

    vin_model.train()
    vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.0001)
    cnn_model.train()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)

    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()

    for epoch in range(config.epochs):
        for i, (reward_map, S1, S2, value, pi) in enumerate(data_loader):
            reward_map = reward_map.to(device)
            S1 = S1.to(device)
            S2 = S2.to(device)
            value = value.to(device)
            pi = pi.to(device)

            # Forward pass

            logits, preds, _ = vin_model(reward_map, S1, S2, config.k)
            V = cnn_model(reward_map, S1, S2)

            l1 = policy_loss(preds, pi)
            l2 = value_loss(V, value)

            loss = l1 + l2

            # Backward pass
            vin_optimizer.zero_grad()
            cnn_optimizer.zero_grad()

            loss.backward()
            vin_optimizer.step()
            cnn_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, iteration {i}, loss: {loss.item()}")


    
def train(config):
    
    eval_window = deque(maxlen=config.eval_window_size)  # Track last 100 episodes
    temp = config.temp_start

    for ep, seed in enumerate(config.seeds):
        print("Starting New Episode:", ep)
     
        R = 0.0
        a_store = []
        # max_steps = 1000
        max_steps = config.max_step 
        steps = 0

        buffer = ReplayBuffer(max_size=config.max_episode_len)
    

        min_obstacles = config.min_obstacles
        max_obstacles = config.max_obstacles
        n = config.n
        num_blocks = config.num_blocks
        square_size = config.square_size

        rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, min_obstacles, max_obstacles, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
        start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)
    
        action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        agent_position = deepcopy(start)
        path = [agent_position]

        print(f"Agent starts at {agent_position}")

        checker = LiveLockChecker(counter=0, last_visited={})

        path_list = [agent_position]


        env = ModifiedGridEnvironment(config, rewards.copy(), obstacles_map, agent_position, goal, living_reward=None, max_steps=1000)
        obs, _ = env.reset()  # Initialize environment with seed
        #puct = PUCT(vin_model, env, agent_position, gamma = 0.99, c_puct=1.44,k=50)
        # puct.vin_model.eval()

        
        while agent_position != goal and steps < max_steps:
            rewards[agent_position] = 0
            
            puct = PUCT(vin_model, cnn_model, env, agent_position, gamma = 0.99, c_puct=1.44,k=50)
            action,new_position = puct.search(agent_position,rewards.copy(),num_simulations=100)
            path.append(new_position)

            state,pi,V, _ = puct.return_results(temp=temp)

            input, S1, S2 = env.get_vin_input(state)
            buffer.add(input, S1, S2, V, pi)

            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)


            # checker.update(agent_position, new_position)b
            # if checker.check(agent_position, new_position):
            #     print("Live Lock Detected in VIN")
            #     table["seed"].append(seed)
            #     table["success"].append(False)  
            #     table["termination"].append("Live Lock")
            #     break
            
            if obstacles_map[new_position[0], new_position[1]]:
                print("Agent moved into an obstacle")
                table["seed"].append(seed)
                table["success"].append(False)
                table["termination"].append("Obstacle")
                break
                
            agent_position = new_position

        temp = max(temp * config.temp_decay, config.temp_end)

        update_models(vin_model,cnn_model,buffer,config)


        eval_window.append(R)


        
def mark_reachable_cells(start, obstacles_map):
    n = obstacles_map.shape[0]
    reachable = np.zeros_like(obstacles_map, dtype=bool)
    queue = deque([start])
    reachable[start] = True

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Down, Right, Up

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not obstacles_map[nx, ny] and not reachable[nx, ny]:
                reachable[nx, ny] = True
                queue.append((nx, ny))

    return reachable




if __name__ == "__main__":
    from fo_solver import visualize_rewards
    from pytorch_value_iteration_networks.model import *
    from types import SimpleNamespace
    import argparse
    from modified_gridenv import ModifiedGridEnvironment
    import pandas as pd
    from eval import LiveLockChecker
    from dl_models import UNetSmall,ValueIterationModel


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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    

    # config = SimpleNamespace(k=50, 
    #                          l_i=2, 
    #                          l_q=4, 
    #                          l_h=150, 
    #                          imsize=20, 
    #                          batch_sz=1,
    #                          )

    vin_model = VIN(config)
    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True,map_location=device)

    vin_model.load_state_dict(vin_weights)
    vin_model.to(device)


    cnn_model = UNetSmall()
    cnn_weights = torch.load("/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/model_weights/smallfinal_model.pt",weights_only=True,map_location=device)
    cnn_model.load_state_dict(cnn_weights)
    
    
    # cnn_model = ValueIterationModel()
    # cnn_model.to(device)

    vin_model.eval()
    cnn_model.eval()
    
    table = defaultdict(list)

    #NOTE: 15893 is an example of a seed that causes a live lock

    for seed in [15893]:

        #seed = 17367
        n = 20
        min_obstacles = 2
        max_obstacles = 10
        n = 20
        num_blocks = 5
        square_size = 10    

        rewards, obstacles_map = init_random_reachable_map(n, "block", num_blocks, 2, 20, obstacle_type="block", square_size=square_size, obstacle_map=None, seed=seed)
        start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)

        reachable = mark_reachable_cells(start, obstacles_map)
        rewards[~reachable] = 0  # Set rewards to zero for unreachable cells
    

        action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        agent_position = deepcopy(start)
        path = [agent_position]

        print(f"Agent starts at {agent_position}")

        checker = LiveLockChecker(counter=0, last_visited={})

        path_list = [agent_position]

        env = ModifiedGridEnvironment(config, rewards.copy(), obstacles_map, agent_position, goal, living_reward=None, max_steps=1000)
        while agent_position != goal:
            rewards[agent_position] = 0
            
            puct = PUCT(vin_model, cnn_model, env, agent_position, gamma = 0.99, c_puct=1.5,k=50,alpha=0.15,epsilon=0)
            action,new_position = puct.search(agent_position,rewards.copy(),num_simulations=100)
            path.append(new_position)

            # checker.update(agent_position, new_position)
            # if checker.check(agent_position, new_position):
            #     print("Live Lock Detected in VIN")
            #     table["seed"].append(seed)
            #     table["success"].append(False)  
            #     table["termination"].append("Live Lock")
            #     break
            
            if obstacles_map[new_position[0], new_position[1]]:
                print("Agent moved into an obstacle")
                table["seed"].append(seed)
                table["success"].append(False)
                table["termination"].append("Obstacle")
                break
                
            agent_position = new_position
            path_list.append(agent_position)

            visualize_rewards(rewards,obstacles_map,agent_position,goal)

            if agent_position == goal:
                print("Agent reached the goal")
                table["seed"].append(seed)
                table["success"].append(True)
                table["termination"].append("Goal Reached")


    

    # df = pd.DataFrame(table)
    # df.to_csv("puct_results.csv",index=False)









        

  
    








