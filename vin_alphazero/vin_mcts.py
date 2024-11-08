import numpy as np
from copy import deepcopy
import random
from collections import defaultdict
import torch
from gridworld_env import GridEnvironment

"""
MCTS with Chance Nodes to handle stochastic environments. This implementation stores the visit and reward values in nodes. .

The state is the cooridnates of the agent in the gridworld, obstacle location and the reward distribution....
"""

class DecisionNode:
    """
    Decision node class, labelled by a state.
    """
    def __init__(self, parent, state, is_terminal, reward, value):
        """
        Args:
            parent (ChanceNode): The parent node of the decision node.
            state (Union[int,np.ndarray]): Environment state.
            weight (float): Probability to occur given the parent (state-action pair)
            is_terminal (bool): Is the state terminal.
            reward (float): immediate reward for reaching this state.

        Attributes:
            children (list): List of child nodes.
            value (float): Value of the state.
            weighted_value (float): Weighted value of the state.

        """
        self.parent = parent

        if isinstance(state, np.ndarray):
            state = tuple(state)
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = {}
        self.children_priors = []
        self.value = value # value of state predicted by the neural network -- equal to reward if terminal
        self.reward = reward# immediate reward
        self.W = 0 
        self.N = 0
        
        self.delta_changes = {} # {(x,y):delta} # store the changes in value for each state.

        def upadate_reward(self,x,y,new_reward):
            self.delta_changes[(x,y)] = new_reward

        def get_full_map(self,iniital_reward_map):
            reward_map = iniital_reward_map.copy()
            for key in self.delta_changes.keys():
                reward_map[key] = self.delta_changes[key]
            return reward_map

        

class ChanceNode:
    """
    Chance node class, labelled by a state-action pair.
    The state is accessed via the parent attribute.
    """
    def __init__(self, parent, action,Q_init):
        """
        Args:
            parent (DecicionsNode): Parent node of the chance node, a decision node.
            action (int): Action taken from the parent node, ie state_1 has child (state_2,action_1) say 
        
        Attributes:
            children (list): List of child nodes (DecisionNode)
            value (float): Value of the state-action pair.
            depth (int): Depth of the node in the tree.
        """
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.value = 0
        self.N = 0 
        self.W = 0
        self.Q = Q_init


def stable_normalizer(x,temp):
    x  = (x/(np.max(x)))**(1/temp)
    return np.abs(x/(np.sum(x)))

class MCTS:
    """MCTS with Chance Nodes for alphazero . Compatible with OpenAI Gym environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simluation is the "default" policy. 
    """
    def __init__(self,env:GridEnvironment,state,model,d,m,c,gamma,alpha=1,epsilon=0,k=50) -> None:
        """
        Args:
            env (gym.Env): The environment to run the MCTS on.
            state (Union[int, np.ndarray]): The state to start the MCTS from.
            model (nn.Module): The neural network model to use for the AlphaZero agent.
            d (int): The depth of the MCTS.
            m (int): The number of simulations to run.
            c (float): The exploration constant.
            gamma (float): The discount factor.

        Attributes:
            v0 (DecisionNode): The root node of the tree.
            possible_actions (list): List of possible actions in the environment.
            Qsa (dict): Dictionary to store Q values for state-action pairs.
            Nsa (dict): Dictionary to store visit counts for state-action pairs.
            Ns (dict): Dictionary to store visit counts for states.

        """
        self.env = env # This is the current state of the mdp
        self.d = d # depth #TODO icorportae this into simulation depth
        self.m = m # number of simulations
        self.c = c # exploration constant
        
        self.gamma = gamma        
        
        # self.Qsa = {}  # stores Q values for s,a pairs, defaults to Qsa of 0
        # self.Nsa = {}  # stores visit counts for s,a pairs, default to Nsa of 0
        # self.Ns = {} # stores visit counts for states, default to Ns of 0
        # self.P = {} # stores the prior probabilities of the actions for a given state
        # self.W = {} # unnormalized cumulative rewards for each state


        ### Deep Learning Model
        self.model = model
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # VIN iteration parameter
        self.k = k 

        self.model = self.model.to(self.device)

        ### ROOT NODE ####
        self.v0 = DecisionNode(parent=None,state=state,is_terminal=False,reward=0,value=0)
        
        with torch.no_grad():
            input, state_x, state_y = self.env.get_vin_input(self.v0.state)
            input = input.to(self.device)
            state_x = state_x.to(self.device)
            state_y = state_y.to(self.device)

            logits,action_probs,values = self.model(input, state_x, state_y, k = self.k)

            val = values[0,0,self.v0.state[0],self.v0.state[1]] # val should be a float. 

        action_probs = action_probs.cpu().numpy().flatten()

        self.possible_actions = [x for x in range(len(action_probs))] #TODO make this more general
        noise = np.random.dirichlet([alpha]*len(action_probs))
        self.v0.value = val
        priors = (1 - epsilon)*action_probs + epsilon*noise
        self.v0.children_priors = priors
        self.v0.children = {a:ChanceNode(parent=self.v0,action=a,Q_init = val) for a in self.env.get_valid_actions(self.v0.state)} #TODO make this more general
        
        # store init reward map for the environment




    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod TODO there may be a better way to do this. 
            expanded_node = self._tree_policy(self.v0) # get the leaf node from the tree policy
            self._backpropagation(expanded_node.value,expanded_node) # update the Q values and visit counts for the nodes in the path from the root to the leaf 


    def return_results(self,temp):
        """Return the results of the MCTS search after it is finished.
        Args:
            temp (float): Parameter for getting the pi values.
        Returns:
            state (np.ndarray): Game state, usually as a 1D np.ndarray.
            pi (np.ndarray): Action probabilities from this state to the next state as collected by MCTS.
            V_target (float): Actual value of the next state.
        """

        counts = np.array([child.N for child in self.v0.children.values()])  
        Q = np.array([child.Q for child in self.v0.children.values()])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q) #calcualte expected value of next state
        best_action = np.argmax(counts)

        if isinstance(self.v0.state, int):
            arr = torch.zeros(self.env.observation_space.n)
            arr[self.v0.state] = 1
            arr = arr.unsqueeze(0)
            state = arr
        else:
            state = self.v0.state # state in this case is the current position of the agent.
 
        return state, pi_target, V_target, best_action

    def _tree_policy(self, node) -> ChanceNode:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            ChanceNode: The leaf node reached by the tree policy.
        """
        depth = 0
        while True:
            # select the best child node based on the UCT value
            action = self._best_action(node) # select step based on UCT value
            chance_node = node.children[action] # select the child node based on the action
            node,backprop = self._expand_chance_node(chance_node) # step through the environment to get the next state if the node exists keep going else return the node. 

            if node.is_terminal or backprop or depth == self.d:
                # return the decision node if is terminal or if it has no children. 
                assert(type(node)==DecisionNode)
                return node
            depth += 1
    
    def _evaluate_decision_node(self,v:DecisionNode):
        """Evaluate the value of a decision node using the neural network model. (Replaces random rollout with a neural network.)

        Args:
            v (DecisionNode): Leaf node to evaluate.

        Returns:
            action_probs (np.ndarray): Action probabilities for all legal moves in from current state.
            val (float): Value of the current state.
        """

        if v.is_terminal:
            return None,v.reward
        

        ## Get VIN predictions... 
        with torch.no_grad():
            input, state_x, state_y = self.sim_env.get_vin_input(v.state)
            input = input.to(self.device)
            state_x = state_x.to(self.device)
            state_y = state_y.to(self.device)
            logits,action_probs,values = self.model(input, state_x, state_y, k = self.k)

            val = values[0,0,v.state[0],v.state[1]] # val should be a float. 



        action_probs = action_probs.cpu().numpy().flatten()
        val = val.cpu().numpy().flatten()
        val = val[0]

        return action_probs,val

    def _expand_decision_node(self,v:DecisionNode,action):
        """Expand the tree by adding a new decision node to the tree.
        """
        assert(type(v) == DecisionNode)
        if v.is_terminal:
            return v
        
        # for a in self.possible_actions:
        #     new_node = ChanceNode(parent=v,action=a)
        #     v.children.append(new_node)

        new_node = ChanceNode(parent=v,action=action)
        v.children.append(new_node)

        return new_node
    
    def _expand_chance_node(self,v:ChanceNode):
        """Expand the tree by adding a new chance node to the tree.

        Args:
            v (ChanceNode): The node to expand.
        Returns:
            new_node (DecisionNode): The new node added to the tree.

        """
        assert(type(v) == ChanceNode)
        action = v.action
        obs,reward,term,_,info = self.sim_env.step(action)

        # existing_child = [child for child in v.children if child.state == obs]
        # if existing_child:
        #     return existing_child[0],False
        # else:

        new_node = DecisionNode(parent=v,state=obs,is_terminal=term,reward=reward,value=0)
        action_probs, value = self._evaluate_decision_node(new_node)
        if term: #If we expand to a terminal state, we set the value to the reward of the terminal state.
            value = reward
        new_node.value = value
        new_node.children_priors = action_probs

        for a in self.possible_actions:
            if a in self.sim_env.get_valid_actions(new_node.state): # make this a set at some point, okay for now because the action space is small
                new_node.children[a]=ChanceNode(parent=new_node,action=a,Q_init=value)
        
        v.children.append(new_node)
        return new_node,True

    def _backpropagation(self,R,v,depth=0):
        """Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """

        assert(type(v) == DecisionNode)
        assert(v.value == R) 

        while v.parent:
            R = v.reward + self.gamma * R
            chance_node = v.parent
            self._update_metrics_chance_node(chance_node,R)
            v = chance_node.parent
            self._update_metrics_decision_node(v)

    def _update_metrics_chance_node(self, chance_node, reward):
        """Update the Q values and visit counts for state-action pairs and states.

        Args:
            state (Union[int,]): _description
            action (_type_): _description_
            reward (_type_): _description_
        """
        assert(type(chance_node) == ChanceNode)

        chance_node.N += 1
        chance_node.W += reward
        chance_node.Q = chance_node.W / chance_node.N

    def _update_metrics_decision_node(self, decision_node):
        """Update the visit counts for states.
        """
        assert(type(decision_node) == DecisionNode)
        decision_node.N += 1
        
    def _best_action(self, v):
        """Select the best action based on the Q values of the state-action pairs.
        Returns:
            best_action (int): the best action to take.
        """
        assert isinstance(v, DecisionNode)
        best_action = None
        best_value = -np.inf

        state = v.state
        possible_actions = self.sim_env.get_valid_actions(state)

        # Filter `children_priors` to match valid actions
        filtered_priors = [v.children_priors[action] for action in possible_actions]

        # Renormalize the priors 
        # Renormalize the priors if they do not sum to 1
        prior_sum = sum(filtered_priors)
        if prior_sum != 1:
            filtered_priors = [prior / prior_sum for prior in filtered_priors]

        for action, prior in zip(possible_actions, filtered_priors):
            ucb_value = v.children[action].Q + self.c * prior * np.sqrt(1 + v.N) / (1 + v.children[action].N)
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action

        if best_action is None:
            raise ValueError("No best action found")
        
        return best_action

    
    def _network_input_checker(self, x):
        """Make sure the input to the neural network is in the correct format
        """

        s = x 
        if type(s) == int:
            arr = torch.zeros(self.env.observation_space.n)
            arr[s] = 1
            arr = arr.unsqueeze(0)
            s = arr

        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(x)

        s = s.to(self.device)
        return s
    

    
if __name__ == "__main__":
    pass
