from mcts import MCTS,DecisionNode,ChanceNode
import yaml
import rtamt
import numpy as np
import random
from copy import deepcopy
import warnings
from collections import namedtuple
"""This is a subclass of the base MCTS implementation where we incorporate STL expressions in to the action selection process
"""

class DistanceCalculator:
    def __init__(self,area_of_interest):
        if not isinstance(area_of_interest,np.ndarray):
            area_of_interest = np.array(area_of_interest)
        self.area_of_interest = area_of_interest

    def _compute_distance(self,current_location):
        if not isinstance(current_location,np.ndarray):
            current_location = np.array(current_location)
        return np.linalg.norm(current_location - self.area_of_interest,1)
    
    def __call__(self, current_location):
        return self._compute_distance(current_location)
    
def always_moving_toward_goal(trace,calculator:DistanceCalculator):
    """To be called during the expansion step 

    Computes the robustnesss degree for the invariant: G_[t_0,t](d(t) - d(t-1) < 0)
    """
    if len(trace) < 2:
        return 0
    
    distances = [calculator(p) for p in trace]
    if distances == []:
        print("uh oh")
    robustness_values = [distances[t-1] - distances[t] for t in range(1, len(distances))]

    return min(robustness_values)

# def negative_distance(current_position, goal_position):
#     return -np.linalg.norm(current_position - goal_position, ord=1)

def negative_distance(trace,calcualtor):
    p = trace[-1]
    distances = calcualtor(p)
    return -distances

class STLMCTS(MCTS):

    def __init__(self,
                 env, 
                 state, 
                 d,
                 m, 
                 c, 
                 gamma, 
                 heuristic=False, 
                 c_heuristic=0,
                 vin=None,
                 puct=False, 
                 temperature=1, 
                 tree_depth=None,
                 seed=None):
        """MCTS with STL action pruning and SLT robusteness degree guided search.
        """
        super().__init__(env, state, d, m, c, gamma, heuristic, vin, puct, temperature, tree_depth,seed=seed)

        # Dictionary that keeps track of state visits...
        self.visits = {}
        self.Hsa = {} # Table of heurstic degree of robustness values. 
        self.c_heuristic = c_heuristic

    def search(self,areas_of_interest=[]):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)
        contstain satisfacoint score... in addition to 

        Args:
            areas_of_interest list(tuple(np.ndarray,float))): A list of tuples that contain areas of interest and weight to put on the

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """

        # If there are areas of interest pay attention to that first

        self.areas_of_interest = areas_of_interest

        if areas_of_interest:
            self.heuristic_calculators = [(DistanceCalculator(aoi),w) for aoi,w in areas_of_interest]
            if sum([x[1] for x in self.heuristic_calculators]) != 1.0:
                warnings.warn("Provided weights do not sum to one, renormalizing")
                #TODO: Renormalize bad weights, 
        else:
            self.heuristic_calculators = [lambda x: 0]

        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            vl = self._tree_policy(self.v0) #vl is the last node visitied by the tree search as chance node
            expanded_node = self._expand(vl) 
            if type(expanded_node) == ChanceNode:
                expanded_node = self._expand(expanded_node) #DecisionNode
                #self.trajectory.append(expanded_node.coord)
            R = self._simulation_policy(expanded_node) #R is the reward from the simulation (default policy)
            self._backpropagation(R,expanded_node)
            
        action_values = [self.Qsa.get((self.v0.state, a), 0) for a in self.possible_actions] # Q values for s a pairs
        visit_counts = [self.Nsa.get((self.v0.state, a), 0) for a in self.possible_actions]
        ba = np.argmax(visit_counts)
        return ba,action_values
     
    def _tree_policy(self, node) -> ChanceNode:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            ChanceNode: The leaf node reached by the tree policy.
        """

        assert isinstance(node, DecisionNode)
        #self.trajectory = [node.coord] #should be appending the location of the root node
        while node.children: #BUG: Returns a list sometimes
            if type(node) == DecisionNode:
                node = self._selection(node)
                assert(type(node) == ChanceNode)
            else: # chance node
                assert(type(node) == ChanceNode)
                node = self._expand(node) 
                assert(type(node) == DecisionNode),f"got {type(node)} instead of DecisionNode"
                assert node.coord is not None
                #self.trajectory.append(node.coord)
        return node
    
    
    def act(self, observation, forward=False):
        return super().act(observation, forward)
    
    def _selection(self, v: DecisionNode):

        #TODO: only compute the heursitic after simulation ... then back propogat that "stl comnformatity value"
        assert isinstance(v,DecisionNode)
        best_value = -np.inf
        best_nodes = []
        children = v.children
        for child in children:
            sa = (child.parent.state, child.action)
            if sa in self.Qsa:
                ucb_value = self.Qsa[sa] + self.c * np.sqrt(np.log(self.Ns.get(sa[0], 1)) / self.Nsa[sa])
                # if self.Hsa.get(sa, 0) == 0:
                #     # Compute the heuristic value using always_moving_toward_goal function
                #     heuristic_value = self._compute_heuristic(child)
                #     # heuristic_value = 0 
                # else:
                heuristic_value = self.Hsa[sa]
                ucb_value += self.c_heuristic * heuristic_value
            else:
                # Use heuristic value for unvisited nodes
                if self.Hsa.get(sa, 0) == 0:
                    # heuristic_value = self._compute_heuristic(child)
                    heuristic_value = 0 
                else:
                    heuristic_value = self.Hsa[sa]
                ucb_value = self.c_heuristic * heuristic_value

            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes) if best_nodes else None
    
    def _compute_heuristic(self,v:DecisionNode):
        """Compute robustness heuristic -- distance based heuristic 
        """
        traj = self._get_trace(v)
        # heuristic_value = sum([always_moving_toward_goal(traj,calculator)*w for calculator,w in self.heuristic_calculators ])

        if self.areas_of_interest:
            heuristic_value = sum([negative_distance(traj,calculator)*w for calculator,w in self.heuristic_calculators ])
        else:
            return 0
        
        return heuristic_value

    def _prune_actions(self,actions):
        raise NotImplementedError
    
    def _get_trace(self,v:DecisionNode):
        assert isinstance(v,DecisionNode)
        trace = []

        v_leaf = v

        while v is not None:
            if isinstance(v,DecisionNode):
                trace.append(v.coord) 
            v = v.parent

        trace.reverse()

        assert trace[0] == self.v0.coord
        assert trace[-1] == v_leaf.coord

        return trace    

    def _read_stl_config_file(self,STL_config:str):
        """Read YAML STL Spec file
        """
        with open(STL_config,"r") as file:
            stl_spec_dict = yaml.safe_load(file)
        
        assert("guidence" in stl_spec_dict.keys())
        assert("pruning" in stl_spec_dict.keys())

        self.guidence_specs = stl_spec_dict["guidence"]
        self.pruning_specs = stl_spec_dict["pruning"]

    def _make_monitor(self):
        self.monitor = rtamt.StlDiscreteTimeSpecification()

        # parse guidence dict

        self.guidence_specs["variables"]

        self.guidence_specs["constants"]

        self.guidence_specs["specs"]        

    def _backpropagation(self, R, v, depth=0):
        """Backpropogate both the value estimate and the robustness metric.
        """

        # assert self.trajectory[0] == self.v0.coord
        # # assert self.trajectory[-1] == v.coord, f"trajectory mismatch got {v.coord}"

        # if self.trajectory[-1]  != v.coord:
        #     warnings.warn("traj mismatch")

        # compute the heuristic score for the trace. How well does the path explore up to this point match our stl specifications? 
        h = self._compute_heuristic(v)

        depth = 0 
        while v:
            v.value += R
            if type(v) == ChanceNode:
                self.update_metrics_chance_node(v.parent.state,v.action,R,h)
            else:
                assert(type(v) == DecisionNode)
                self.update_metrics_decision_node(v.state)
            # R = R*(self.gamma**depth)

            depth+=1
            v = v.parent

    def update_metrics_chance_node(self, state, action, reward,h):
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
            self.Hsa[sa] = (self.Hsa[sa] * (self.Nsa[sa]-1) + h) / self.Nsa[sa]

        else:
            self.Qsa[sa] = reward
            self.Nsa[sa] = 1
            self.Hsa[sa] = h

    def act(self, observation,areas_of_interest=[],forward=False):
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
            self.Hsa = {}


        best_action, _ = self.search(areas_of_interest=areas_of_interest)

        return best_action

    

        


if __name__ == "__main__":
    from experiments.experiment_setup import *
    from mcts import MCTS

    config = read_config("/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/experiments/configs/static_env_baseline_mcts_5x5.yaml")
    env = make_env(27,config)

    reward_list = []

    for seed in range(25):

        obs,_ = env.reset(seed=seed)
        intit_states = env.get_state()

        obstacles_map = intit_states["obstacles"]

        mcts = STLMCTS(env,obs,d=100,m=500,c=1.44,gamma=0.999,c_heuristic=0.5,seed=seed)
        #mcts = MCTS(env,obs,d=100,m=500,c=1.4,gamma=0.999)

        step = 0 
        collisions = 0  
        done = False
        # This is an upper bound on the size of the state space.
        # mcts = MCTS(env,observation,d=100,m=500,c=5,gamma=0.9)
        max_steps  = 100
        total_reward = 0
        start = time.time()

        areas_of_interest = [((4,4),1.0)]

        while step < max_steps and not done:
            visualize_rewards(env.unwrapped.current_rewards,env.unwrapped.obstacles,env.unwrapped.agent_position,(4,4))

            # mcts = ns_gym.benchmark_algorithms.MCTS(env,observation,d=25,m=100,c=1,gamma=0.999)
            # assert mcts.root.state == observation, "Root state must match observation!"

            action = mcts.act(obs,areas_of_interest=areas_of_interest, forward=False)
            # action = mcts.act(obs)
            obs,reward,done,_,info = env.step(action)

            # Check if the agent has reached an area of interest
            current_position = obs[1]
            areas_of_interest = [(aoi, w) for aoi, w in areas_of_interest if not np.array_equal(current_position, aoi)]

            


            if info["collision"]:
                collisions += 1
        
            total_reward += reward
            step += 1

            print(f"\rStep count {step}",end="",flush=True)

        
        print("reward: ",total_reward)
        reward_list.append(total_reward)

    print("Mean reward: ", np.mean(reward_list))





