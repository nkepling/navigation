import gymnasium as gym
from copy import deepcopy
import random
from collections import defaultdict
import math
import torch

"""
MCTS Implemenation. This implementation uses a global table to store the Q values and visit counts for state-action pairs and states. Compatible with Gymnasium environments.
"""

class Node:
    """
    Node class, labeled by a (state,actions) pair.
    """
    def __init__(self, parent, state, action, is_terminal, reward):
        """
        Args:
            parent (Node): The parent node of this node.
            state (Union[int,np.ndarray,tuple]): Environment state.
            action (int): Action taken to reach this state.
            is_terminal (bool): Is the state terminal.
            reward (float): Immediate reward for reaching this state.

        Attributes:
            children (list): List of child nodes. Index is the action taken to reach the child node.

        """
        self.parent = parent # parent node
        self.state = state # state of the environment (observation)
        self.action = action # action taken to reach this state
        self.is_terminal = is_terminal # is the state terminal (done)
        self.reward = reward # immediate reward for reaching this state
        self.children = [] # list of child nodes, initially empty, index is the action taken to reach the child node

    def is_leaf(self):
        return not self.children
    
    def is_root(self):
        return self.parent is None
    
    def is_terminal(self):
        return self.is_terminal






class MCTS:
    """Vanilla MCTS; Compatible with Gymnasium environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simluation is the "default" policy. 

        For reference see:
        
        A Survey of Monte Carlo Tree Search Methods Browne et al. 2012
        
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6145622


        You will implement the following methods:

        1. search()
        2. _tree_policy()
        3. _default_policy()
        4. _selection()
        5. _expand()
        6. _backpropagation()
        7. update_node()
        8. best_child()
        9. best_action()


        NOTE: To be compatable with NS-Gym be sure to pass all observations and rewards into the type_checker() util function. 

        For example:

        ```python
        observation, reward, done, truncated,info = env.step(action)
        observation, reward = type_checker(observation, reward)

        ###################### or ######################

        observation, _  = type_checker(observation, None)

        ###################### or ######################

        _, reward = type_checker(None, reward)
        ```


    """
    def __init__(self,vin_model,cnn_model,env,state,d,m,c,gamma,alpha,epsilon) -> None:
        """
        Args:
            env (gym.Env): The environment to run the MCTS on.
            state (Union[int, np.ndarray]): The state to start the MCTS from.
            d (int): The rollout depth of the MCTS.
            m (int): The number of simulations to run.
            c (float): The exploration constant.
            gamma (float): The discount factor.

        Attributes:
            root (Node): The root node of the tree.
            possible_actions (list): List of possible actions in the environment.
            Qsa (dict): Dictionary to store Q values for state-action pairs.
            Nsa (dict): Dictionary to store visit counts for state-action pairs.
            Ns (dict): Dictionary to store visit counts for states.

        """


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.env = env # gym environment
        self.d = d # rollout depth 
        self.m = m # number of simulations
        self.c = c # UCT exploration constant
        
        self.possible_actions = [x for x in range(env.action_space.n)] # possible actions in the environment
        self.gamma = gamma # discount factor
        self.Qsa = defaultdict(float)  # stores Q values for s,a pairs, defaults to Qsa of 0
        self.Nsa = defaultdict(float)  # stores visit counts for s,a pairs, default to Nsa of 0
        self.Ns = defaultdict(float) # stores visit counts for states, default to Ns of 0
        # Root node

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma



        self.vin_model = vin_model
        self.vin_model.to(self.device)
        self.cnn_model = cnn_model
        self.cnn_model.to(self.device)
    
        state,_ = type_checker(state,None)
        self.root = Node(None,state,None,False,0) 

    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action with the most visits.

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deepcopy of of the original environment for simulation. 
            # YOUR CODE HERE
            leaf_node = self._tree_policy(self.root)
            expand_node = self._expand(leaf_node)   
            R = self._default_policy(expand_node)
            self._backpropagation(R,expand_node)

        best_action = self.best_action(self.root)
        action_values = [self.Qsa[(self.root.state,a)] for a in self.possible_actions]
        return best_action,action_values


    def _tree_policy(self, node:Node):
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node or terminal state.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            node: The leaf node reached by the tree policy.
        """
        # YOUR CODE HERE
        while node.children:
            node = self._selection(node)
            observation,reward,done,truncated,info = self.sim_env.step(node.action)
        return node

    def _default_policy(self,node:Node):
        """Simulate/Playout step 
        While state is non-terminal and choose an action uniformly at random, transition to new state. Return the reward for final state. 

        Args:
            node (Node): The node to start the simulation from.
        """

        # YOUR CODE HERE

        if node.is_terminal:
            return node.reward
        
        tot_reward = 0
        terminated = False
        truncated = False
        depth = 0

        while not terminated and not truncated and depth < self.d:
            action = random.choice(self.possible_actions)
            observation,reward,terminated,truncated,info = self.sim_env.step(action)
            observation,reward = type_checker(observation,reward)
            tot_reward += reward*self.gamma**depth
            depth += 1

        return tot_reward

    def _selection(self,node:Node):
        """Pick the next node to go down in the search tree based on UTC formula.
        """
        # YOUR CODE HERE

        best_value = -math.inf
        best_nodes = []
        children = node.children
        for child in children:

            if self.Nsa[(child.state,child.action)] == 0:
                value = math.inf
            else:
                value = self.Qsa[(child.state,child.action)] + self.c * math.sqrt(math.log(self.Ns[node.state])/self.Nsa[(child.state,child.action)])


            if value > best_value:
                best_value = value
                best_nodes = [child]
            elif value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes) if best_nodes else None

    def _expand(self,node:Node):
        """Expand the tree by adding a new node to the tree.
        """
        # YOUR CODE HERE

        if node.is_terminal:
            return node
        
        for a in self.possible_actions:
            temp_env = deepcopy(self.sim_env)
            observation,reward,done,truncated,info = temp_env.step(a)
            observation,reward = type_checker(observation,reward)
            new_node = Node(node,observation,a,done,reward)
            node.children.append(new_node)

        return random.choice(node.children)
    
    def _evaluate(self,observation):
        """Evaluate the state using the VIN model.
        """
        # YOUR CODE HERE
        observation = torch.tensor(observation).to(self.device)
        observation = observation.unsqueeze(0)
        value = self.vin_model(observation)
            

    def _backpropagation(self,R,node:Node):
        """Backtrack to update the number of times a node has beenm visited and the value of a node until we reach the root node. 
        """
        # YOUR CODE HERE
        depth = 0
        while node:
            R = R*(self.gamma**depth)
            self.Qsa[(node.state,node.action)]  = (self.Qsa[(node.state,node.action)] * self.Nsa[(node.state,node.action)] + R)/(self.Nsa[(node.state,node.action)]+1)
            self.Nsa[(node.state,node.action)] += 1
            self.Ns[node.state] += 1
            node = node.parent
            depth += 1

    def best_action(self,node:Node):
        """Get the best action to take from the root node based on visit counts.
        """
        #YOUR CODE HERE

        children = node.children

        best_action = max(children,key = lambda x: self.Nsa[(x.state,x.action)])

        return best_action.action









if __name__ == "__main__":
    pass



