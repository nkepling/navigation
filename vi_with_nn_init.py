from fo_solver import value_iteration, extract_policy, visualize_policy_and_rewards, visualize_rewards, pick_start_and_goal
from utils import *
from nn_training import *
from copy import deepcopy
import matplotlib.pyplot as plt


def nn_initialized_vi(model,n, rewards, obstacles, gamma,neighbors,threshold=1e-6):

    V = model(reformat_input(rewards, obstacles)).detach().numpy().squeeze()
    # Initialize the value function
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if obstacles[i, j]:
                    continue  # Skip obstacle cells
                v = V[i, j]
                # Get the possible next states and rewards
                next_states = []
                v_hat = []
                for s in neighbors[(i, j)]:
                    v_hat.append(V[s[0], s[1]])

                # Bellman update
                V[i, j] = rewards[i, j] + gamma * max(v_hat, default=0)
                delta = max(delta, abs(v - V[i, j]))
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    pass