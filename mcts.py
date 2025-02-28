import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg

import ns_gym.base as base
import random
from collections import defaultdict
import torch


# from nn_training import reformat_input

"""
MCTS with Chance Nodes to handle stochastic environments. This implementation used a global table to store the Q values and visit counts for state-action pairs and states. Compatible with OpenAI Gym environments.
"""




"""Nodes for tree
"""

class DecisionNode: 
    """
    Decision node class, labelled by a state.
    """
    def __init__(self, parent, state, weight, is_terminal,reward):
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
        self.weight = weight  # Probability to occur
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.value = 0 # value of state
        self.reward = reward# immediate reward
        self.weighted_value = self.weight * self.value

        self.coord = state[1]




class ChanceNode:
    """
    Chance node class, labelled by a state-action pair.
    The state is accessed via the parent attribute.
    """
    def __init__(self, parent, action):
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

class MCTS:
    """Vanilla MCTS with Chance Nodes. Compatible with OpenAI Gym environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simluation is the "default" policy. 
    """
    def __init__(self,env:gym.Env,state,d,m,c,gamma,heuristic=False,vin=None,puct=False,temperature=1.0,tree_depth=None,seed=None,device=None,k=None) -> None:
        """
        Args:
            env (gym.Env): The environment to run the MCTS on.
            state (Union[int, np.ndarray]): The state to start the MCTS from.
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
        self.d = d # depth 
        self.m = m # number of simulations
        self.c = c # exploration constant
        self.tree_depth = tree_depth

        self.vin = vin

        self.puct = puct
        self.temperature = temperature

        if self.vin:
            self.k = k
            assert k != None, "k must be provided"
            vin.eval()
            if device:
                self.device = device
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            vin.to(self.device)

        # set random seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.v0 = DecisionNode(parent=None,state=state,weight=1,is_terminal=False,reward=0)

        if not isinstance(env.action_space,gym.spaces.Discrete):
            raise ValueError("Only discrete action spaces are supported")
        
        self.possible_actions = [x for x in range(env.action_space.n)]
        self.gamma = gamma        
        self.Qsa = {}  # stores Q values for s,a pairs, defaults to Qsa of 0
        self.Nsa = {}  # stores visit counts for s,a pairs, default to Nsa of 0
        self.Ns = {} # stores visit counts for states, default to Ns of 0

        self.heuristic = heuristic  
        self.vin_cache = {}

    def search(self, **kwargs):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            vl = self._tree_policy(self.v0) #vl is the last node visitied by the tree search as chance node
            expanded_node = self._expand(vl) 
            if type(expanded_node) == ChanceNode:
                expanded_node = self._expand(expanded_node) #DecisionNode
            R = self._simulation_policy(expanded_node) #R is the reward from the simulation (default policy)
            self._backpropagation(R,expanded_node)



        # action_values = [self.Qsa[(self.v0.state,a)] for a in self.possible_actions] # Q values for s a pairs
        # visit_counts = [self.Nsa[(self.v0.state,a)] for a in self.possible_actions]


        action_values = [self.Qsa.get((self.v0.state, a), 0) for a in self.possible_actions] # Q values for s a pairs
        visit_counts = [self.Nsa.get((self.v0.state, a), 0) for a in self.possible_actions]
        # hueristic_scores = [self.Hsa.get((self.v0.state,a),0) for a in self.possible_actions]

        ba = np.argmax(visit_counts)
            
        # print(f"Visit counts: {visit_counts}")

        # print(f"Action values: {action_values}")    
        # print(f"Best action: {ba}")
        # print("sum of visit counts",sum(visit_counts))



        # ba = self.best_action(self.v0)
        # ba = np.argmax(action_values)
        
        # ba = np.argmax(v)


        return ba,action_values

    def _tree_policy(self, node) -> ChanceNode:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            ChanceNode: The leaf node reached by the tree policy.
        """
        while node.children: 
            if type(node) == DecisionNode:
                node = self._selection(node)
                assert(type(node) == ChanceNode)
            else: # chance node
                assert(type(node) == ChanceNode)
                node = self._expand(node) 
                assert(type(node) == DecisionNode),f"got {type(node)} instead of DecisionNode"


        return node
    
    def _simulation_policy(self,v:DecisionNode):

        if self.heuristic:
            _,R = self._vin_policy(v)
            # R = self._vin_rollout(v)
            return R
    
        return self._default_policy(v)
    
    def _vin_policy(self,v:DecisionNode):
        """Use the Value Iteration Network to get the value of the state.
        """
        assert isinstance(v,DecisionNode)
        # check if 



        state_dict = self.sim_env.unwrapped.get_state()
        # key = tuple(state_dict)

        # if key in self.vin_cache.keys():
        #     return self.vin_cache[key]


        input,x,y = self._reformat_input(state_dict["rewards"],state_dict["obstacles"],state_dict["agent_position"])
        logits,probs,value = self.vin(input,x,y,k=self.k)
        x = state_dict["agent_position"][0]
        y = state_dict["agent_position"][1]        
        R = value[:, 0, x, y].item()
        probs = probs.cpu().detach().numpy().squeeze()

        # self.vin_cache[key] = (probs,R)

        return probs,R
    
    def _vin_rollout(self,v:DecisionNode):
        """Use the trajectory from the value iteration network to do a rollout rather than the value estimate

        Essentially from a newly expande node simply grab the trajectory that the the VIN would do to some depth d.  The return the discoutned cummulative reward. 
        """

        
        assert isinstance(v,DecisionNode)
        depth = 0
        vin_trajectory = [] 
        while depth < self.d:
            state_dict = self.sim_env.unwrapped.get_state()
            input,x,y = self._reformat_input(state_dict["rewards"],state_dict["obstacles"],state_dict["agent_position"])
            logits,probs,value = self.vin(input,x,y,k=self.k)
            x = torch.tensor(state_dict["agent_position"][0])
            y = torch.tensor(state_dict["agent_position"][1])    
            
            probs = probs.detach().numpy().squeeze()

            a = np.random.choice(self.possible_actions,p=probs)
            # a = np.argmax(probs)
            observation,reward,terminated,truncated,info = self.sim_env.step(a)

            vin_trajectory.append(reward*self.gamma**depth)

            if terminated or truncated:
                break
            
            depth+=1

        R = sum(vin_trajectory)
        return R

    
    def _reformat_input(self,rewards,obstacles,coords):
        """Reformat the input for the NN model
        """
        temp = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
        # obstacles_map  = np.where(obstacles_map,-1,0)
        obstacles_map = torch.tensor(obstacles, dtype=torch.float32).unsqueeze(0)
        input = torch.cat((temp, obstacles_map), dim=0)
        input = input.unsqueeze(0)

        n  = len(obstacles)
        assert input.shape == (1,2,n,n)

        x = torch.tensor(coords[0])
        y = torch.tensor(coords[1])


        if self.device != "cpu":
            input = input.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)

        return input,x,y


    def _default_policy(self,v:DecisionNode):
        """Simulate/Playout step 
        While state is non-terminal choose  an action uniformly at random, transition to new state. Return the reward for final  state. 

        Args:
            v (DecisionNode): The node to start the simulation from.
        """
        if v.is_terminal:
            return v.reward
        tot_reward = 0
        terminated = False
        truncated = False
        depth = 0
        while not terminated and depth < self.d and not truncated:
            action = np.random.choice(self.possible_actions)
            observation,reward,terminated,truncated,info = self.sim_env.step(action)
            tot_reward += reward*self.gamma**depth
            depth+=1
        return tot_reward

    def _selection(self,v:DecisionNode):
        """Pick the next node to go down in the search tree based on UTC formula.
        """

        if self.puct:
            best_child = self._puct(v)
        else:
            best_child = self.best_child(v)
        return best_child

    def _expand(self,node):
        """Expand the tree by adding a new node to the tree. Handles both decision and chance nodes.
        """
        
        if type(node) == DecisionNode:
            if node.is_terminal:
                return node
            
            if self.tree_depth is not None and node.depth >= self.tree_depth:
                print("Reached tree depth")
                return node
            
            for a in range(self.sim_env.action_space.n):
                new_node = ChanceNode(parent=node,action=a)
                node.children.append(new_node)
            return np.random.choice(node.children) 

        else: # chance node
            action = node.action
            assert(type(node)==ChanceNode)
            obs,reward,term,_,info = self.sim_env.step(action)
            existing_child = [child for child in node.children if child.state == obs]
            if existing_child:
                return existing_child[0]
            else:
                if "prob" in info:
                    w = info["prob"]
                else:
                    w = 1    
                new_node = DecisionNode(parent=node,state=obs,weight=w,is_terminal=term,reward=reward)
                node.children.append(new_node)
                return new_node

    def _backpropagation(self,R,v,depth=0):
        """Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """
        depth = 0 
        while v:
            v.value += R
            if type(v) == ChanceNode:
                self.update_metrics_chance_node(v.parent.state,v.action,R)
            else:
                assert(type(v) == DecisionNode)
                self.update_metrics_decision_node(v.state)
            # R = R*(self.gamma**depth)

            depth+=1
            v = v.parent

    def update_metrics_chance_node(self, state, action, reward):
        """Update the Q values and visit counts for state-action pairs and states.

        Args:
            state (Union[int,]): _description
            action (_type_): _description_
            reward (_type_): _description_
        """

        if isinstance(state, np.ndarray):
            state = tuple(state)

        if isinstance(action, np.ndarray):
            action = tuple(action)
        sa = (state, action)


          # Increment visit count first to prevent using stale value
        if sa in self.Nsa:
            self.Nsa[sa] += 1
            self.Qsa[sa] = (self.Qsa[sa] * (self.Nsa[sa] - 1) + reward) / self.Nsa[sa]
        else:
            self.Qsa[sa] = reward
            self.Nsa[sa] = 1


        # if sa in self.Qsa:
        #     self.Qsa[sa] = (self.Qsa[sa] * self.Nsa[sa] + reward) / (self.Nsa[sa] + 1)
        #     self.Nsa[sa] += 1
        # else:
        #     self.Qsa[sa] = reward
        #     self.Nsa[sa] = 1

    def update_metrics_decision_node(self, state):
        """Update the visit counts for states.
        """
        if state in self.Ns:
            self.Ns[state] += 1
        else:
            self.Ns[state] = 1

    
    def best_child(self,v):
        """Find the best child nodes based on the UCT value.

        This method is only called for decision nodes.

        Args:
            exploration_constant (_type_, optional): _description_. Defaults to math.sqrt(2).

        Returns:
            Node: The best child node based on the UCT value.
            action: The action that leads to the best child node.
        """
        
        best_value = -np.inf
        best_nodes = []
        children = v.children
        for child in children:
            sa = (child.parent.state, child.action)
            if sa in self.Qsa:
                ucb_value = self.Qsa[sa] + self.c * np.sqrt(
                    np.log(self.Ns.get(sa[0], 1)) / self.Nsa[sa])
            else:
                ucb_value = self.c * np.sqrt(
                    np.log(self.Ns.get(sa[0], 1)) / 1)  # Assume at least one visit
                ucb_value = np.inf

            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes) if best_nodes else None
    
    def _puct(self, v: DecisionNode):
        """Select the best child node based on the PUCT formula."""
        children = v.children

        if not children:
            return None
        
        # Sum of visits for all actions from this state
        sum_visits_s = 0
        for child in children:
            sa = (v.state, child.action)
            sum_visits_s += self.Nsa.get(sa, 0)
        
        best_value = -float('inf')
        best_children = []

        policy_prior = self._vin_policy(v)[0]

        # If you have a different prior, e.g. from a policy network, replace this part:
        
        if self.temperature is not None and self.temperature != 1.0:
            # Option A: "Raise to 1/temperature" and re-normalize
            policy_prior = policy_prior ** (1.0 / self.temperature)
            # Re-normalize (avoid division by zero if policy_prior sums to 0)
            sum_p = policy_prior.sum()
            if sum_p > 0:
                policy_prior /= sum_p
            else:
                # If everything was zero, fall back to uniform
                policy_prior = np.full_like(policy_prior, 1.0 / len(policy_prior))

        for child in children:
            sa = (v.state, child.action)
            
            # Q(s,a) defaulting to 0 if unseen
            q_val = self.Qsa.get(sa, 0.0)
            # N(s,a) defaulting to 0 if unseen
            n_sa = self.Nsa.get(sa, 0)
            
            # If total visits from this state is 0, treat sum_visits_s as 1 to avoid sqrt(0).
            # (Or just skip the child if sum_visits_s=0, but typically you do a small constant.)
            if sum_visits_s == 0:
                sum_visits_s = 1
            
            # PUCT exploration term
            u_val = self.c * policy_prior[child.action] * np.sqrt(sum_visits_s) / (1 + n_sa)
            
            puct_val = q_val + u_val
            
            if puct_val > best_value:
                best_value = puct_val
                best_children = [child]
            elif np.isclose(puct_val, best_value):
                best_children.append(child)
        
        best_child = random.choice(best_children)
        return best_child
    


    def _STL_selection(self,v:DecisionNode):
        """Select the next node to go down in the search tree based on the STL robustness value agumented with the UCT formula.
        """
        raise NotImplementedError
    
    def best_action(self,v):
        """Select the best action based on the Q values of the state-action pairs.
        Returns:
            best_action(int): best action to)
        """
        best_action = None
        best_avg_value = -np.inf

        s = v.state # root is Type[Node] 

        # Iterate through all possible actions from this state
        for a in range(self.env.action_space.n):
            sa = (s, a)  # Create a state-action pair
            # Check if this state-action pair has been explored
            if sa in self.Qsa and sa in self.Nsa and self.Nsa[sa] > 0:
                # avg_value = self.Qsa[sa] / self.Nsa[sa]  # Calculate average value
                avg_value = self.Qsa[sa]
                if avg_value > best_avg_value:
                    best_avg_value = avg_value
                    best_action = a

        # Ensure a valid action is selected, even if no action has been explored
        if best_action is None and self.possible_actions:
            best_action = np.random.choice(self.possible_actions)

        return best_action
    
    def act(self, observation, forward=False):
        """
        Decide on an action using the MCTS search, reinitializing the tree structure.

        Args:
            observation (Union[int, np.ndarray]): The current state or observation of the environment.

        Returns:
            int: The selected action.
        """

        # Reinitialize the instance by calling __init__
        # self.__init__(env, observation, self.d, self.m, self.c, self.gamma)

        # Perform MCTS search to determine the best action


        if forward:
            if not self.v0.children:
                self.v0 = DecisionNode(parent=None, state=observation, weight=1, is_terminal=False,reward=0)
            else:
                for child in self.v0.children:
                    for c in child.children:
                        if c.state == observation:
                            self.v0 = c
                            break
        else:
            self.v0 = DecisionNode(parent=None, state=observation, weight=1, is_terminal=False,reward=0)
            self.Ns = {}
            self.Nsa = {}
            self.Qsa = {}


        best_action, _ = self.search()

        return best_action


if __name__ == "__main__":
    #from modified_gridenv import ModifiedGridEnvironment
    from new_grid_env import GridworldEnv,WrapForMCTS
    # from gridworld_env import GridEnvironment
    from utils import init_random_reachable_map, pick_start_and_goal
    from fo_solver import visualize_rewards
    import argparse
    import gymnasium as gym
    import time

    import torch
    from pytorch_value_iteration_networks.model import *
    from experiments.experiment_setup import *
    from mcts import MCTS
    from pytorch_value_iteration_networks.model import *
    from types import SimpleNamespace
    import torch


    parser = argparse.ArgumentParser()

    ### Gridworld parameters

    parser.add_argument('--n', type=int, default=5, help='Grid size')
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
    parser.add_argument('--k', type=int, default=16, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_sz', type=int, default=1, help='Batch size')

    config = parser.parse_args()
    
    #n = 20# size of the grid
    # n = 10
    # n = 6
    n= 5

    # n = 10
    min_obstacles = 2 # minimum number of obstacles
    max_obstacles = 3 # maximum number of obstacles

    max_steps = 100 # maximum number of steps to take
    step = 0 

    # seed = 200
    seed = 89


    # seed = 42
    
    # q
    # seed = 199
    config = read_config("/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/experiments/configs/static_env_baseline_vin_stl_mcts_5x5.yaml")
    
    seed = config["env_seed"]
    env = make_env(seed,config)

    table = {"seed":[],"reward":[],"steps":[],"time":[],"collisions":[],"found_all_rewards":[],"max_steps":[]}

    start_time = time.time()
    for i in range(27):
        # rewards,obstacles_map = init_random_reachable_map(n, 
        #                             "block", 
        #                             min_obstacles, 
        #                             max_obstacles, 
        #                             obstacle_type="block", 
        #                             obstacle_map=None, 
        #                             seed=seed,
        #                             num_reward_blocks=(2,5),
        #                             reward_square_size=(1,2),
        #                             obstacle_cluster_prob=0.0,
        #                             obstacle_square_sizes=(1,2))
        
        # rewards[0,0] = 0




        # rewards = rewards * 100 

        # rewards = np.zeros(shape=(n,n))

        # rewards[2,9] = 0.3

        # rewards[6,4] = 0.1
        # rewards[6,3] = 0.1
        # rewards[5,4] = 0.1
        # rewards[5,3] = 0.1

        # rewards[8,7] = 0.3

        # rewards[4,1] = 0.2
        # rewards[4,4] = 0.8
        # rewards[4,5] = 1
        # rewards[2,5] = 0.2
        # rewards[4,2] = 1
        # rewards[2,1] = 0.5
        # # rewards[3,4] = 0.5
        # rewards[10,5] = 10

        # rewards[5,10] = 10

        # print(rewards)
        # print(obstacles_map)

 




        config = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # device = "mps"

        vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/pytorch_value_iteration_networks/trained/vin_5x5.pth', weights_only=True, map_location=device)
        #vin_weights  = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/pytorch_value_iteration_networks/trained/vin_full_traj.pth', weights_only=True, map_location=device)
        vin = VIN(config)

        vin.load_state_dict(vin_weights)

        vin.to(device)
        vin.eval()
        
        
        # start, goal = pick_start_and_goal(rewards, obstacles_map,seed=seed)

        # #visualize_rewards(rewards,obstacles_map,start,goal)
        # # env = ModifiedGridEnvironment(config,rewards,obstacles_map,start,goal,living_reward=-0.1,shuffle=False,train=False,max_steps=1000)
        # env = GridworldEnv(rewards,obstacles_map,start,goal,living_reward=0.0)
        # env = WrapForMCTS(env)

        print(env.get_state_space_size())

        #env = gym.make("FrozenLake-v1",is_slippery=True,render_mode="ansi")

        r = []

        total_reward = 0
        observation, _ = env.reset()
        mcts = MCTS(env,observation,d=20,m=500,c=1.4,gamma=0.9,puct=True,temperature=0.8,vin=vin,heuristic=True,k=16,device="mps")

        step = 0
        collisions = 0  
        done = False
        # This is an upper bound on the size of the state space.
        # mcts = MCTS(env,observation,d=100,m=500,c=5,gamma=0.9)

        while step < max_steps and not done:
            visualize_rewards(env.unwrapped.current_rewards,env.unwrapped.obstacles,env.unwrapped.agent_position,(4,4))


            # mcts = ns_gym.benchmark_algorithms.MCTS(env,observation,d=25,m=100,c=1,gamma=0.999)
            # assert mcts.root.state == observation, "Root state must match observation!"

            action = mcts.act(observation, forward=False)
            observation,reward,done,_,info = env.step(action)

            if info["collision"]:
                collisions += 1
        
            total_reward += reward
            step += 1

            print(f"\rStep count {step}",end="",flush=True)

        
        print("reward: ",total_reward)

    print("Total Time, ", time.time() - start_time)


    




