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

def softmax(logits):
    """Compute softmax values for the given logits."""
    max_logit = np.max(logits)  # For numerical stability
    exp_logits = np.exp(logits - max_logit)  # Subtract max logit to prevent overflow
    return exp_logits / np.sum(exp_logits)

class UCTNode:
    def __init__(self, coords = (0,0), prior_prob=0, parent=None,action=None,depth=0,val_init=0):
        self.visits = 0  # N(s,a) in PUCT
        self.init_value = 0 
        self.value_sum = 0  # W(s,a) - cumulative value
        self.prior_prob = prior_prob  # P(s,a) - prior probability from VIN logits
        self.parent = parent
        self.children = {}  # dictionary of actions to UCTNodes
        self.coords = coords  # coordinates of the state in the grid
        self.depth = 0 # depth of the node in the tree
        self.action = None # action taken to reach this node

        self.val_init =  val_init

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

    def expand(self, action_probs, valid_next_states,next_state_value):
        """Expand the node by creating child nodes for each possible action."""

        for action, prob in action_probs.items():
            if action in valid_next_states:
                child_coords = valid_next_states[action]
                self.children[action] = UCTNode(coords=child_coords, prior_prob=prob, parent=self,action=action,depth=self.depth+1,val_init=next_state_value[action])

    def is_leaf(self):
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0
    
class PUCT:
    def __init__(self, vin_model, env:GridEnvironment, start, gamma, c_puct=1.0,k=50):
        self.vin_model = vin_model  # Value Iteration Network to get values and logits
        self.env = env  # The grid environment instance
        self.c_puct = c_puct
        self.root = UCTNode(coords=start)  # Root node of the tree
        self.gamma = gamma
        self.k = k 


    def search(self,num_simulations):
        """Run multiple simulations to build the tree.
        satrt: the start state of the agent (coords)
        """
        for _ in range(num_simulations):
            node = self.root
            env.reset(node.coords)
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
    def _get_best_action(self,node):
        """Select the best action based on the visit counts."""
        best_puct_value = -np.inf
        best_action = None
        best_child = None

        for action, child in node.get_children().items():
            puct_value = child.get_value(self.c_puct)

            if puct_value > best_puct_value:
                    best_puct_value = puct_value
                    best_action = action
                    best_child = child

        return best_action

    def _select(self, node):
        """Select the child node with the highest PUCT value."""
        best_action = self._get_best_action(node)
        self.env.step(best_action)
        return node.get_children()[best_action]

    def _expand_and_evaluate(self, node):
        """Expand the node using the VIN model and return the value."""
        # Run the VIN model to get values and logits for the current state
        input,state_x,state_y = self.env.get_vin_input(node.coords)

        assert input.shape == (1,2,self.env.n,self.env.n), f"input shape is {input.shape}"
        logits, preds, value = self.vin_model(input,state_x,state_y,self.k)

        # convert from torch tensor to numpy array
        logits = logits.detach().numpy()
        preds = preds.detach().numpy()
        value = value.detach().numpy() # this is a value map ... 

        # Expand the node with the action probabilities
        valid_actions = self.env.get_valid_actions(node.coords)



        next_state_value = self._get_next_state_value(node,value,valid_actions) # dict of next state values for each action



        # value of state associated with leaf node
        # val = value[0,0,node.coords[0],node.coords[1]]

        # if len(valid_actions) != len(preds):
        #     # assume there is always at least one valid action
        #     if len(valid_actions) == 0:
        #         raise ValueError("No valid actions available.")
            
        #     else:
        #         # If the number of valid actions is less than the number of actions in the VIN model, 
        #         # we need to filter the predictions to only include valid action

        #         # set invalid action probabilities to 0
        #         logits = np.where(np.isin(np.arange(4), valid_actions), logits, -np.inf)

        #         preds = softmax(logits).flatten()
    
        action_probs = preds.flatten()
        action_prob_dict = {i: action_probs[i] for i in range(len(action_probs))}
        node.expand(action_prob_dict, self.env.get_valid_next_states(node.coords),next_state_value)

        return next_state_value

    def _backpropagate(self, node):
        """Backpropagate the value to the root.
        value is  
        """

        # best_value = max(value.values()) # back propagate the best value out of all actions 

        value = node.val_init

        while node is not None:
            node.update(value)
            value *= self.gamma
            node = node.parent

    def _get_next_state_value(self, node, value, valid_actions):
        """Given value map grab next state value for current agent position

        Returns:
            next_state_value (dict): value of the next state for each each action
        """
        action_state_dict = self.env.get_valid_next_states(node.coords)

        values = np.zeros(self.env.get_action_space_size())

        for a in range(self.env.get_action_space_size()):
            
            if a not in valid_actions:
                values[a] = -np.inf
            else:
                values[a] = value[0,0,action_state_dict[a][0],action_state_dict[a][1]]
        
        next_state_value = {a: values[a] for a in range(self.env.get_action_space_size())}

        return next_state_value
    
    def _update_root(self, best_action):
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

    config = SimpleNamespace(k=50, l_i=2, l_q=4, l_h=150, imsize=20, batch_sz=1)

    vin_model = VIN(config)
    seed = 100
    n = 20
    rewards,obstacles_map = init_random_reachable_map(n, "block", 3, 2, 20, obstacle_type="block", square_size=10, obstacle_map=None, seed=seed)
    start, goal = pick_start_and_goal(rewards, obstacles_map, seed=seed)
   

    action_to_dir = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    agent_position = deepcopy(start)
    path = [agent_position]


    while agent_position != goal:
        rewards[agent_position] = 0
        env = GridEnvironment(20, rewards.copy(), obstacles_map,agent_position,goal)
        puct = PUCT(vin_model, env, agent_position, gamma = 0.8, c_puct=1.14)
        action,new_position = puct.search(num_simulations=500)
        path.append(new_position)
        print(f"Agent moved to {new_position}")
        visualize_rewards(rewards,obstacles_map,agent_position,goal)
        agent_position = new_position
     

  
    








