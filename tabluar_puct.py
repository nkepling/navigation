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

from collections import defaultdict
import hashlib


def softmax(logits):
    """Compute softmax values for the given logits."""
    max_logit = np.max(logits)  # For numerical stability
    exp_logits = np.exp(logits - max_logit)  # Subtract max logit to prevent overflow
    return exp_logits / np.sum(exp_logits)



def encode_state(env, state_coords):
        """Encode the state and reward map as a hashable key."""
        rewards = env.get_rewards()
        hasher = hashlib.md5()
        hasher.update(rewards.tobytes())
        hasher.update(str(state_coords).encode('utf-8'))
        return hasher.hexdigest()

class UCTNode:
    def __init__(self, state_hash, reward, coords = (0,0), prior_prob=0, parent=None,action=None,depth=0,val_init=0):
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

        self.state_hash = state_hash

    def update(self, value):
        """Update the node with new value."""
        self.visits += 1
        self.value_sum += value

    def get_value(self, c_puct):
        """Calculate the PUCT value for this node."""
        u_value = c_puct * self.prior_prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        q_value = self.value_sum / (1 + self.visits)  # Q(s,a) = W(s,a) / N(s,a)
        return q_value + u_value

    def get_children(self):
        """Return all child nodes."""
        return self.children
    
    def expand(self, action_probs, valid_next_states, next_state_value, next_state_rewards, env):
        """Expand the node by creating child nodes for each possible action."""
        for action, prob in action_probs.items():
            if action in valid_next_states:
                child_coords = valid_next_states[action]
                assert isinstance(child_coords, tuple), f"child_coords is {child_coords}"
                reward = next_state_rewards[action]  # next state immediate reward
                # Generate a unique hash for the child state
                state_hash = encode_state(env, child_coords)
                # Create a child node with the unique state hash
                self.children[action] = UCTNode(
                    state_hash=state_hash,
                    reward=reward,
                    coords=child_coords,
                    prior_prob=prob,
                    parent=self,
                    action=action,
                    depth=self.depth + 1,
                    val_init=next_state_value[action]
                )
    def is_leaf(self):
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0
    
class tabPUCT:
    def __init__(self, vin_model, env:GridEnvironment, start, gamma, c_puct=1.0, k=50,alpha=0.15,epsilon=0.25):
        self.vin_model = vin_model  # Value Iteration Network to get values and logits
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

        self.gamma = gamma
        # add dirichlet noise to the prior probabilities to encourage exploration

        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = defaultdict(float)  # Q(s,a) - cumulative value/running average
        self.N = defaultdict(int)  # N(s,a) - visit count
        self.W = defaultdict(float)  # W(s,a) - cumulative value
        self.P = defaultdict(float)  # P(s,a) - prior probability from VIN logits


    def _update_tables(self, state_coords, action, value):
        """Update Q, N, and W tables for a given state-action pair."""
        state_action_hash = (encode_state(self.env,state_coords), action)
        self.N[state_action_hash] += 1
        self.W[state_action_hash] += value
        self.Q[state_action_hash] = self.W[state_action_hash] / self.N[state_action_hash]
    

    def search(self,root_coords,init_reward,num_simulations):
        """Run multiple simulations to build the tree.
        satrt: the start state of the agent (coords)
        """
        logits,preds,value = self._evaluate(UCTNode(state_hash=encode_state(self.env,root_coords),reward=0,coords=root_coords))
        val_init = value[0,0,root_coords[0],root_coords[1]]
        self.root = UCTNode(state_hash=encode_state(self.env,root_coords),reward=0,coords=root_coords,val_init=val_init)  # Root node of the tree

        for _ in range(num_simulations):
            node = self.root
            self.env.reset(node.coords,init_reward)
            # Traverse the tree until we reach a leaf node
            while not node.is_leaf():
                node = self._select(node)

            # Expand and evaluate the leaf node
            value = self._expand_and_evaluate(node)

            # Backpropagate the value through the tree
            self._backpropagate(node)

        # best_action = self._get_best_action(self.root)

        best_action = max(self.root.get_children(), key=lambda x: self.root.get_children()[x].visits)

        best_child = self.root.get_children()[best_action]

        return best_action, best_child.coords
    
    def _get_best_action(self, node):
        best_action = None
        best_puct_value = -np.inf

        for action, child in node.get_children().items():
            state_action_hash = (node.state_hash, action)
            puct_value = self.Q[state_action_hash]

            # penalize multiple visitd actions actions

            penalty = 0.1 * self.N[state_action_hash]
            puct_value -= penalty


            if puct_value > best_puct_value:
                best_puct_value = puct_value
                best_action = action

        return best_action

    def _select(self, node):
        """Select the child node with the highest PUCT value."""
        best_action = self._get_best_action(node)
        self.env.step(best_action)
        return node.get_children()[best_action]
    
    def _evaluate(self, node):
        input,state_x,state_y = self.env.get_vin_input(node.coords)

        input = input.to(self.device)
        state_x = state_x.to(self.device)
        state_y = state_y.to(self.device)


        assert input.shape == (1,2,self.env.n,self.env.n), f"input shape is {input.shape}"
        logits, preds, value = self.vin_model(input,state_x,state_y,self.k)

        # convert from torch tensor to numpy array
        logits = logits.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        value = value.cpu().detach().numpy() # this is a value map ...

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
            action_probs = (1-self.epsilon) * action_probs + self.epsilon * np.random.dirichlet([self.alpha] * len(action_probs))
        ##############################################################################################################################

        action_prob_dict = {i: action_probs[i] for i in range(len(action_probs))}
        reward_map = self.env.get_rewards()

        next_states_rewads = {a: reward_map[valid_next_state] for a, valid_next_state in self.env.get_valid_next_states(node.coords).items()}

        node.expand(action_prob_dict, self.env.get_valid_next_states(node.coords),next_state_value,next_states_rewads,self.env)

        return next_state_value

    def _backpropagate(self, node):
            """Backtrack to update the values and visits of each node up to the root."""
            R = node.val_init
            while node.parent:
                # Calculate the cumulative reward for the current state-action pair
                R = node.reward + self.gamma * R

                # Get the state-action hash for the current node and its parent action
                state_action_hash = (node.parent.state_hash, node.action)

                # Update visit count (N), cumulative value (W), and Q value
                self.N[state_action_hash] += 1
                self.W[state_action_hash] += R
                self.Q[state_action_hash] = self.W[state_action_hash] / self.N[state_action_hash]

                # Move to the parent node
                node.update(R)
                node = node.parent

            # Update the root node visit count as well
            self.N[(self.root.state_hash, None)] += 1

        
    def _get_next_state_value(self, node, value, valid_actions):
        """Given value map grab next state value for current agent position

        Returns:
            next_state_value (dict): value of the next state for each each action
        """
        action_state_dict = self.env.get_valid_next_states(node.coords)

        values = np.zeros(4)

        for a in range(4):
            
            if a not in valid_actions:
                values[a] = -np.inf
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





if __name__ == "__main__":
    from fo_solver import *
    from pytorch_value_iteration_networks.model import *
    from types import SimpleNamespace
    import argparse

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

    vin_model.eval()


    seed = 100
    n = 20
    rewards,obstacles_map = init_random_reachable_map(n, "block", 3, 2, 20, obstacle_type="block", square_size=10, obstacle_map=None, seed=seed)
    start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)
   

    action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    agent_position = deepcopy(start)
    path = [agent_position]


    while agent_position != goal:
        rewards[agent_position] = 0
        env = GridEnvironment(config, rewards.copy(), obstacles_map,agent_position,goal)
        puct = PUCT(vin_model, env, agent_position, gamma = 1, c_puct=1.44,k=50)
        action,new_position = puct.search(num_simulations=100)
        path.append(new_position)
        print(f"Agent moved to {new_position}")
        visualize_rewards(rewards,obstacles_map,agent_position,goal)
        agent_position = new_position
     

  
    








