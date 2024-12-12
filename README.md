## Files:

1. fo_solver.py: File to implement fully observable Value Iteration by Backward Induction (Bellman, 1957).
2. q_learning.py: File to implement tabular Q-Learning.
3. utils.py: utility file to initialize map, sample start and goal, etc.
4. pytorch_value_iteration_networks/models.py: contains a value iteration network it is slightly modified from here (https://github.com/kentsommer/pytorch-value-iteration-networks)
5. pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth: Are the trained neural network weights
6. run_single_episode.py: contains an example of how to call the neural network and explore the map.
7. generate_data_for_vin.py: Is the script used to generate training data from the VIN


## Details about the VIN

The VIN model take in a config object to set all of its hyper parameters. See run_single_episode.py for an example of how that is constructed. 

The `k` parameter is the number of times the out put of the VIN ruurses through the network. 

**INPUTS:**

A 2 x 20 x 20 image and the coordinates of the agent

**OUTPUS***

A 1 x 4 vector coresponding to action logits.  Actions are mapped as `actions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # up, right, down, left `





## To run dynaic programming solution:

Run fo_solver.py 

Change initial configurations as desired in the fo_solver.py file:

1. n (side of the total n x n grid)
2. config (distribution of positive probability cells, currently set to block, meaning positive regions appear as large squares)
3. num_blocks (number of positive probability blocks)
4. num_obstacles (number of obstacles)
5. obstacle_type (type of obstacles, currently set to block, same as above)
6. square_size (length of reward blocks)
7. gamma (discount factor for the MDP)
8. random_map (set to True if you want to generate a random map for each run). If set to False, it uses the map stored in mdp_data.pkl.

