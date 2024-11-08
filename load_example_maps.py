import numpy as np
import matplotlib.pyplot as plt
from utils import *
from fo_solver import *
import pickle
import os
from eval import get_vi_path
from scipy.ndimage import label, center_of_mass





sample_arr_dir = "/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/sample_arrays/"

drone_height = 10
n = 400 

cluster_sizes_list = []

# Loop through each file in the directory
for example in os.listdir(sample_arr_dir):
    full_path = os.path.join(sample_arr_dir, example)
    
    with open(full_path, 'rb') as f:
        sample_arr = pickle.load(f)
    
    # Extract obstacles map
    obstacles = sample_arr['obstacleMap_array'] >= drone_height
    no_fly_zones = sample_arr['obstacleMap_array'] == 50

    obstacles = np.logical_xor(obstacles, no_fly_zones)

    rewards = sample_arr['targetMap_array']

    print(set(rewards.flatten()))


    print(np.sum(rewards))

    fov_cells  = get_fov((100,100),(100,101),obstacles,(400,400))

    # for cell in fov_cells:
    #     rewards[cell] = 1

    visualize_rewards(rewards, obstacles,start=(0,0),goal=(n-1,n-1))
    # Label clusters in the obstacle map
    labeled_array, num_clusters = label(obstacles)

    # Calculate the size of each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude baqckground (label 0)
    cluster_sizes_list.extend(cluster_sizes)

    plt.figure(figsize=(10, 6))

    print("Number of clusters: ", len(cluster_sizes_list))
    plt.hist(cluster_sizes, bins=50, edgecolor='black')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cluster Sizes')
    plt.grid(True)
    plt.show()


    # start_time = time.time()
    # neighbors = precompute_next_states(n, obstacles)
    # # Viter = value_iteration(n, rewards, obstacles, gamma,neighbors)
    # Viter = value_iteration_with_extened_fov(n, rewards, obstacles, gamma,neighbors)
    # policy = extract_policy(Viter, obstacles,neighbors,n=n)
    # end_time = time.time()
    # print("Time to solve: ", end_time - start_time)





# Plotting the histogram of cluster sizes



    # start_time = time.time()
    # Viter = value_iteration(n, rewards, obstacles, gamma,neighbors)
    # policy = extract_policy(Viter, obstacles,neighbors,n=n)
    # end_time = time.time()
    # print("Time to solve: ", end_time - start_time)


    # Solve the problem



