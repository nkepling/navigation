import numpy as np
from copy import deepcopy
import random
from collections import defaultdict
import torch
from pytorch_value_iteration_networks.model import VIN

"""
MCTS with unified nodes to handle stochastic environments. This implementation stores the visit and reward values in nodes, and uses a VIN model for node evaluation. The search is depth-limited as there are no terminal states.
"""

class Node:
    """
    Node class, labelled by a state and action.
    """
    def __init__(self, parent, state, action, weight, reward, value):
        """
        Args:
            parent (Node): The parent node of the current node.
            state (Union[int,np.ndarray]): Environment state.
            action (int): Action taken from the parent node to reach this node.
            weight (float): Probability to occur given the parent (state-action pair)
            reward (float): Immediate reward for reaching this state.

        Attributes:
            children (list): List of child nodes.
            value (float): Value of the state.
            action (int): Action taken to reach this node from the parent.
            N (int): Visit count.
            W (float): Cumulative reward.
            Q (float): Mean value of the state-action pair.
        """
        self.parent = parent
        self.state = state
        self.action = action
        self.weight = weight  # Ground truth probability to occur given the parent (state-action pair) -- given by the environment
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.children_priors = []
        self.value = value # value of state predicted by the neural network
        self.reward = reward # immediate reward
        self.W = 0 
        self.N = 0
        self.Q = 0


def stable_normalizer(x, temp):
    x  = (x / (np.max(x))) ** (1 / temp)
    return np.abs(x / (np.sum(x)))

class MCTS:
    """MCTS with unified nodes for AlphaZero using VIN. Compatible with OpenAI Gym environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simulation is the "default" policy. 
    """
    def __init__(self, env, state, model, d, m, c, gamma, alpha=1, epsilon=0) -> None:
        """
        """
        self.env = env # This is the current state of the mdp
        self.d = d # depth limit
        self.m = m # number of simulations
        self.c = c # exploration constant
        self.gamma = gamma        

        ### ROOT NODE ####
        self.v0 = Node(parent=None, state=state, action=None, weight=1, reward=0, value=0)
        action_probs, val = self._evaluate_node(self.v0)

        noise = np.random.dirichlet([alpha] * len(action_probs))
        self.v0.value = val
        priors = (1 - epsilon) * action_probs + epsilon * noise
        self.v0.children_priors = priors
        self.v0.children = [Node(parent=self.v0, state=None, action=a, weight=1, reward=0, value=val) for a in self.possible_actions]

    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)

        Returns:
            best_action(int): best action to take
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            expanded_node = self._tree_policy(self.v0) # get the leaf node from the tree policy
            self._backpropagation(expanded_node.value, expanded_node) # update the Q values and visit counts for the nodes in the path from the root to the leaf 

    def return_results(self, temp):
        """Return the results of the MCTS search after it is finished.
        Args:
            temp (float): Parameter for getting the pi values.
        Returns:
            state (np.ndarray): Game state, usually as a 1D np.ndarray.
            pi (np.ndarray): Action probabilities from this state to the next state as collected by MCTS.
            V_target (float): Actual value of the next state.
        """
        counts = np.array([child.N for child in self.v0.children])  
        Q = np.array([child.Q for child in self.v0.children])
        pi_target = stable_normalizer(counts, temp)
        V_target = np.sum((counts / np.sum(counts)) * Q) # calculate expected value of next state
        best_action = np.argmax(counts)
        
        if isinstance(self.v0.state, int):
            arr = torch.zeros(self.env.observation_space.n)
            arr[self.v0.state] = 1
            arr = arr.unsqueeze(0)
            state = arr
        else:
            state = self.v0.state
        
        return state, pi_target, V_target, best_action

    def _tree_policy(self, node) -> Node:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node, respecting the depth limit.
        Args:
            node (Node): The root node of the tree.
        Returns:
            Node: The leaf node reached by the tree policy.
        """
        depth = 0
        while True:
            action = self._best_action(node) # select step based on UCT value
            child_node = node.children[action] # select the child node based on the action
            node, backprop = self._expand_node(child_node) # step through the environment to get the next state if the node exists keep going else return the node.

            if backprop or depth == self.d:
                assert(isinstance(node, Node))
                return node
            depth += 1
    
    def _evaluate_node(self, v: Node):
        """Evaluate the value of a node using the neural network model. (Replaces random rollout with a neural network.)

        Args:
            v (Node): Leaf node to evaluate.

        Returns:
            action_probs (np.ndarray): Action probabilities for all legal moves in from current state.
            val (float): Value of the current state.
        """
        with torch.no_grad():
            s = v.state

            if isinstance(s, base.Observation):
                s = s.state

            if type(s) == int:
                arr = torch.zeros(self.env.observation_space.n)
                arr[s] = 1
                arr = arr.unsqueeze(0)
                s = arr

            if not isinstance(s, torch.Tensor):
                s = torch.Tensor(s)

            s = s.to(self.device)
            action_probs, val = self.model(s)

        action_probs = action_probs.cpu().numpy().flatten()
        val = val.cpu().numpy().flatten()[0]

        return action_probs, val

    def _expand_node(self, v: Node):
        """Expand the tree by adding a new node to the tree.

        Args:
            v (Node): The node to expand.
        Returns:
            new_node (Node): The new node added to the tree.
        """
        assert(isinstance(v, Node))
        action = v.action
        obs, reward, _, _, info = self.sim_env.step(action)
        obs, reward = self.type_checker(obs, reward)

        existing_child = [child for child in v.children if child.state == obs]
        if existing_child:
            return existing_child[0], False
        else:
            if "prob" in info:
                w = info["prob"]
            else:
                w = 1    

            new_node = Node(parent=v, state=obs, action=None, weight=w, reward=reward, value=0)
            action_probs, value = self._evaluate_node(new_node)
            new_node.value = value
            new_node.children_priors = action_probs

            for a in self.possible_actions:
                new_node.children.append(Node(parent=new_node, state=None, action=a, weight=1, reward=0, value=value))
            
            v.children.append(new_node)
            return new_node, True

    def _backpropagation(self, R, v, depth=0):
        """Backtrack to update the number of times a node has been visited and the value of a node until we reach the root node. 
        """
        assert(isinstance(v, Node))
        assert(v.value == R) 

        while v.parent:
            R = v.reward + self.gamma * R
            self._update_metrics_node(v, R)
            v = v.parent

    def _update_metrics_node(self, node, reward):
        """Update the Q values and visit counts for the node.

        Args:
            node (Node): The node to update metrics for.
            reward (float): The reward value to update.
        """
        assert(isinstance(node, Node))

        node.N += 1
        node.W += reward
        node.Q = node.W / node.N

    def type_checker(self, observation, reward):
        """Converts the observation and reward from base.Observation and base.Reward type to the correct type if they are not already.

        Args:
            observation (_type_): Observation to convert.
            reward (_type_): Reward to convert.

        Returns:
            (int,np.ndarray): Converted observation.
            (float): Converted reward.
        """
        if isinstance(observation, base.Observation):
            observation = observation.state
        if isinstance(observation, np.ndarray):
            observation = tuple(observation)
        if isinstance(reward, base.Reward):
            reward = reward.reward
        return observation, reward
    
    def _best_action(self, v):
        """Select the best action based on the Q values of the state-action pairs.
        Returns:
            best_action(int): Best action to take.
        """
        assert(isinstance(v, Node))
        best_action = None
        best_value = -np.inf

        state = v.state

        for action, prior in zip(self.possible_actions, v.children_priors):
            ucb_value = v.children[action].Q + self.c * prior * np.sqrt(1 + v.N) / (1 + v.children[action].N) 
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action

        return best_action

if __name__ == "__main__":
    pass
