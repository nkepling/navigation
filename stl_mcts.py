from mcts import MCTS,DecisionNode,ChanceNode
import yaml
import rtamt
import numpy as np
import random
from copy import deepcopy
import warnings

from invariant_functions import *

"""This is a subclass of the base MCTS implementation where we incorporate STL expressions in to the action selection process
"""


  
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
                 seed=None,
                 device=None,
                 k=None,
                 rho_min=0.1,
                 rho_high=1.0,
                 alpha_increase=0.1,
                 alpha_decrease=0.1,
                 check_frequency=None,
                 spec_function_list=None):
        """MCTS with STL action pruning and SLT robusteness degree guided search.
        """
        super().__init__(env, state, d, m, c, gamma, heuristic, vin, puct, temperature, tree_depth,seed=seed,device=device,k=k)

        # Dictionary that keeps track of state visits...
        self.visits = {}
        self.Hsa = {} # Table of heurstic degree of robustness values. 
        self.c_heuristic = c_heuristic

        self.rho_min = rho_min
        self.rho_high = rho_high
        self.alpha_increase = alpha_increase
        self.alpha_decrase = alpha_decrease

        self.obstacles_detector = ObstacleDectector(env)

        self.tree_depth = tree_depth

        if check_frequency is None:
            self.check_frequency = 1
        else:
            self.check_frequency = check_frequency

        self.spec_function_list = spec_function_list

        self.history = [] 


    def search(self):
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

        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            vl = self._tree_policy(self.v0) #vl is the last node visitied by the tree search as chance node
            expanded_node = self._expand(vl) 
            if type(expanded_node) == ChanceNode:
                expanded_node = self._expand(expanded_node) #DecisionNode
                #self.trajectory.append(expanded_node.coord)
            R = self._simulation_policy(expanded_node) #R is the reward from the simulation (default policy)
            self._backpropagation(R,expanded_node)

            if k%self.check_frequency == 0 and k > 0:
                trace = self._get_best_trace()
                h = self._compute_heuristic(self.v0,trace)
                self._adapt_policy(h)

    
            
        action_values = [self.Qsa.get((self.v0.state, a), 0) for a in self.possible_actions] # Q values for s a pairs
        visit_counts = [self.Nsa.get((self.v0.state, a), 0) for a in self.possible_actions]
        hueristic_scores = [self.Hsa.get((self.v0.state,a),0) for a in self.possible_actions]

        final_selection_values = [self.Qsa.get((self.v0.state, a), 0) + self.c_heuristic*self.Hsa.get((self.v0.state,a),0) for a in self.possible_actions]


        # ba = np.argmax(final_selection_values)

        # print(5*"#")

        # print("Action vals ", action_values)
        # print("Heuristic vals", hueristic_scores)


        ba = None
        best_value = -np.inf

        for a,val in enumerate(final_selection_values):

            if visit_counts[a] > 0:
                if val > best_value:
                    best_value = val 
                    ba = a 

        # print(final_selection_values)

        ba = np.argmax(visit_counts)
        return ba,final_selection_values
     
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
    
    
    # def act(self, observation, forward=False):
    #     return super().act(observation, forward)
    
    def _get_best_trace(self):
        depth = 0
        trace = []

        trace.append(self.v0.coord)

        if self.tree_depth:
            max_depth = self.tree_depth
        else:
            max_depth = 100

        v = self.v0

        while depth < max_depth:
            if isinstance(v, DecisionNode):
                best_value = -np.inf
                best_child = None

                if not v.children:
                    break

                for child in v.children:
                    sa = (v.state, child.action)
                    q_val = self.Qsa.get(sa, 0.0)
                    h_val = self.Hsa.get(sa, 0.0)
                    n_val = self.Nsa.get(sa,0.0)

                    value = q_val + self.c_heuristic * h_val

                    if value > best_value and n_val > 0:
                        best_value = value
                        best_child = child

                if best_child is None:
                    break

                v = best_child
                
            else: # chance node
                if v.children:
                    v = v.children[0] # determenistic setting so it shoudl only have one child
                    trace.append(v.coord)
                
            depth += 1

        return trace
    
    def _selection(self, v: DecisionNode):


         #TODO: only compute the heursitic after simulation ... then back propogat that "stl comnformatity value"
        assert isinstance(v,DecisionNode)
        best_value = -np.inf
        best_nodes = []
        children = v.children

        if self.puct:
            
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

            # #TODO: Refine this for noise adding
            noise = np.random.uniform(low=0, high=0.1, size=policy_prior.shape)
            policy_prior = policy_prior + noise
            policy_prior /= policy_prior.sum()



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

                heuristic_value  = self.Hsa.get(sa,0.0)
                
                # If total visits from this state is 0, treat sum_visits_s as 1 to avoid sqrt(0).
                # (Or just skip the child if sum_visits_s=0, but typically you do a small constant.)
                if sum_visits_s == 0:
                    sum_visits_s = 1
                
                # PUCT exploration term
                u_val = self.c * policy_prior[child.action] * np.sqrt(sum_visits_s) / (1 + n_sa)
                
                puct_val = q_val + u_val + self.c_heuristic * heuristic_value
                
                if puct_val > best_value:
                    best_value = puct_val
                    best_nodes = [child]
                elif np.isclose(puct_val, best_value):
                    best_nodes.append(child)
            
                
        else:
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
                        #heuristic_value = self._compute_heuristic(child)
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
    

    def _compute_heuristic(self,v:DecisionNode,traj=None):
        """Compute robustness heuristic -- distance based heuristic 
        """

        if traj is None:
            traj = self._get_trace(v)

        if self.spec_function_list is None:
            raise ValueError
        
        if self.history:
            traj = self.history + traj
        
        heuristic_value = conjunction_of_specs(traj,self.spec_function_list)

        return heuristic_value
    

    def _prune_actions(self,actions):
        raise NotImplementedError
    
    def _get_trace(self,v:DecisionNode):
        assert isinstance(v,DecisionNode)
        trace = []

        node_list = []

        v_leaf = v

        while v is not None:
            if isinstance(v,DecisionNode):
                trace.append(v.coord) 
            node_list.append(v)
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
            # h = h*(self.gamma)

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

    def act(self, observation,forward=False):
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


        best_action, _ = self.search()

        return best_action
    
    def _adapt_policy(self, rho):
        """
        Adjust self.temperature based on STL robustness rho.
        Optionally set self.add_noise to True if rho < rho_min.
        """
        if rho < self.rho_min:
            # Increase temperature => More exploration
            delta = self.rho_min - rho
            self.temperature  = min(self.temperature * np.exp(self.alpha_increase * delta) ,2)
            #self.c_heuristic += self.alpha_increase * delta
            self.c_heuristic = min(self.c_heuristic * (1 + self.alpha_increase),5)
            self.add_noise = True
            # print(f"Value updated  ",)
        elif rho > self.rho_high:
            # Decrease temperature => More exploitation
            delta = rho - self.rho_high
            self.temperature = max(self.temperature * np.exp(-self.alpha_decrease * delta),0.8)
            self.c_heuristic  = max(1,self.c_heuristic * (1-self.alpha_decrase))
            self.add_noise = False


    def update_history(self,coord):
        self.history.append(coord)


if __name__ == "__main__":
    from experiments.experiment_setup import *
    from mcts import MCTS
    from pytorch_value_iteration_networks.model import *
    from types import SimpleNamespace
    import torch

    config = read_config("/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/experiments/configs/static_env_baseline_vin_stl_mcts_5x5.yaml")
    
    seed = config["env_seed"]
    env = make_env(seed,config)

    reward_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/navigation/pytorch_value_iteration_networks/trained/vin_5x5.pth', weights_only=True, map_location=device)

    vin =  VIN(SimpleNamespace(**config))

    vin.load_state_dict(vin_weights)

    vin.to(device)
    vin.eval()

    #TODO: add noise to prob priods...
    
    obs,_ = env.reset(seed=seed)
    intit_states = env.get_state()

    obstacles_map = intit_states["obstacles"]


    # spec1 = AvoidCells([(2,2)])

    # spec2 = VistCells([(2,4),(3,4),(4,4)])

    spec2 = VistCells([(4,4),(4,1)])

    spec1 = TemporalWindowSpec(0,7,[(2,2)])

    # spec3 = VisitInOrder([(4,4),(4,1)])

    spec_list = [spec1,spec2]

    mcts = STLMCTS(env,obs,d=100,m=100,c=1.44,gamma=0.9,c_heuristic=1.44,temperature=1,seed=seed,vin=vin,puct=True,k=16,check_frequency=2,spec_function_list=spec_list)
    #mcts = MCTS(env,obs,d=100,m=500,c=1.4,gamma=0.999)
    

    step = 0 
    collisions = 0  
    done = False
    # This is an upper bound on the size of the state space.
    # mcts = MCTS(env,observation,d=100,m=500,c=5,gamma=0.9)
    max_steps  = 100
    total_reward = 0
    start = time.time()

    areas_of_interest = [((4,1),1.0)]


    
    # areas_of_interest = []qq


    start = time.time()
    while step < max_steps and not done:
        # visualize_rewards(env.unwrapped.current_rewards,env.unwrapped.obstacles,env.unwrapped.agent_position,(4,4))

        action = mcts.act(obs,forward=False)
        obs,reward,done,_,info = env.step(action)

        # Check if the agent has reached an area of interest
        current_position = obs[1]

        mcts.update_history(current_position)

        print(mcts.history)

        if info["collision"]:
            collisions += 1
    
        total_reward += reward
        step += 1

        print(f"\rStep count {step}",end="",flush=True)



    print("reward: ",total_reward)
    reward_list.append(total_reward)

    print("c_end ", mcts.c_heuristic)

    print("time ", time.time()-start)










