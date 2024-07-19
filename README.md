Files:

1. fo_solver.py: File to implement fully observable Value Iteration by Backward Induction (Bellman, 1957).
2. q_learning.py: File to implement tabular Q-Learning.
3. utils.py: utility file to initialize map, sample start and goal, etc.

To run:

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

