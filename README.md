Files:

1. fo_solver.py: File to implement fully observable Value Iteration by Backward Induction (Bellman, 1957).
2. q_learning.py: File to implement tabular Q-Learning.
3. utils.py: utility file to initialize map, sample start and goal, etc.

To run:

Run fo_solver.py 

Change initial configurations as desired in the fo_solver.py file:

n (side of the total n x n grid)
config (distribution of positive probability cells, currently set to block, meaning positive regions appear as large squares)
num_blocks (number of positive probability blocks)
num_obstacles (number of obstacles)
obstacle_type (type of obstacles, currently set to block, same as above)
square_size (length of reward blocks)
gamma (discount factor for the MDP)

