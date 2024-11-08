import mctx
import jax
import jax.numpy as jnp
import argparse
from tqdm import tqdm
import pickle
import torch
import numpy as np 
from collections import deque

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from torch import optim 
from torch import nn
from torch.nn import functional as F

from replaybuffer import ReplayBuffer
from torch.utils.data import DataLoader

from pytorch_value_iteration_networks.model import VIN
from vin_mcts import MCTS

from types import SimpleNamespace

from torch.utils.tensorboard import SummaryWriter
from gridworld_env import GridEnvironment



def train_network(buffer, net, config):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set network to train mode
    net.train()

    # Ensure batch_size is set in config
    batch_size = config.batch_size if hasattr(config, 'batch_size') else 32  # Default to 32 if not in config

    # Set up DataLoader with the specified batch size
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True)

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.epochs):
        avg_loss, num_batches = 0.0, 0

        for reward_maps, S1s, S2s, values, pis in dataloader:
            # Move data to the device
            reward_maps = reward_maps.to(device)
            S1s = S1s.to(device)
            S2s = S2s.to(device)
            values = values.to(device)
            pis = pis.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, predictions, vals = net(reward_maps, S1s, S2s, config.k)

            # Calculate loss
            loss = criterion(outputs, pis)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update metrics
            avg_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss /= num_batches
        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.4f}")

class VINAlphaZeroAgent:
    def __init__(self,
                config
                ) -> None:
        """
        Arguments:
            Env: 
            model: AlphaZeroNetwork : Neurla network,f(obs) the outputs v and phi
            mcts: MCTS
            lr : float : learning rate for NN
            n_hidden_layers: int: number of hidden layers in model
            n_hidden_units: int: number of units in each layer of model
            n_episodes: int: Number of training episodes (number ot times we do mcts + model training)
            max_episode_len : int: The maximum number of environment steps before we terminate
        """

        # init model
        self.model = VIN(config)
    
        ######## SET DEVICE ########
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        ######### LOAD MODEL ########
        if config.model_checkpoint_path:
            checkpoint = torch.load(config.model_checkpoint_path,map_location=self.device)
            self.model.load_state_dict(checkpoint)
    
        self.model = self.model.to(self.device)


        ########## MCTS INIT + HYPERPARAMS ########
        self.mcts = MCTS 
        self.gamma = config.gamma
        self.c = config.c
        self.num_mcts_simulations = config.num_mcts_simulations
        self.max_mcts_search_depth = config.max_mcts_search_depth
        self.alpha = config.alpha
        self.epsilon = config.epsilon


    def train(self, config,env:GridEnvironment):
        """
        Overview:
            Trains the agent by iteratively performing MCTS and neural network training.
            Stores episode returns for evaluation.
        """

        self.Env = env

        # Store episode returns
        episode_returns = []
        t_total = 0  # Total number of steps
        R_best = -np.inf

        writer = SummaryWriter("runs/" + config.experiment_name)

        # If no seeds provided, generate them
        if not hasattr(config, 'seeds') or not config.seeds:
            config.seeds = np.random.choice(range(1000000), config.n_episodes, replace=False)

        eval_window = deque(maxlen=config.eval_window_size)  # Track last 100 episodes
        temp = config.temp_start

        for ep, seed in enumerate(config.seeds):
            print("Starting New Episode:", ep)
            obs, _ = self.Env.reset()  # Initialize environment with seed
            R = 0.0
            a_store = []
            
            buffer = ReplayBuffer(max_size=config.max_episode_len)

            self.model.eval()
            for t in range(config.max_episode_len):

                if t % 100 == 0:
                    print("Step: ", t)
                # MCTS step
                mcts = self.mcts(self.Env, state=obs, model=self.model, d=config.max_mcts_search_depth,
                                    m=config.num_mcts_simulations, c=config.c, gamma=config.gamma,
                                    alpha=config.alpha, epsilon=config.epsilon)
                mcts.search()
                state, pi, V, _ = mcts.return_results(temp=temp)

                # Get input for VIN and store state, pi, V in buffer
                input, S1, S2 = self.Env.get_vin_input(state)
                buffer.add(input, S1, S2, V, pi)

                # Take a step in the environment
                a = np.random.choice(len(pi), p=pi)
                a_store.append(a)

                obs, r, done, truncated, _ = self.Env.step(a)
                R += r

                if done:
                    print("DONE ---- Episode Reward:", R)
                    break
                elif truncated:
                    print("Truncated ---- Episode Reward:", R)
                    print("Episode Length:", t)
                    break
                
            # Update temperature
            temp = max(temp * config.temp_decay, config.temp_end)
            eval_window.append(R)

            if len(eval_window) == config.eval_window_size:
                mean_reward_over_last_100_ep = np.mean(eval_window)
                if mean_reward_over_last_100_ep > R_best:
                    print("Mean reward:", mean_reward_over_last_100_ep)
                    mod_name = config.experiment_name + f'_best_model_checkpoint.pth'
                    torch.save(self.model.state_dict(), mod_name)
                    R_best = mean_reward_over_last_100_ep

            # Save model checkpoint periodically
            if ep % 10 == 0:
                print("Saving model checkpoint")
                print(f"Episode {ep}, Mean Reward: {np.mean(episode_returns[-100:])}")
                torch.save(self.model.state_dict(), config.experiment_name + f'ep_{ep}_model_checkpoint.pth')

            writer.add_scalar("Episode Rewards", R, ep)
            episode_returns.append(R)
            print("Training Model")
            print(f"Iter {ep}")
            
            # Train network with data in buffer
            train_network(buffer, self.model, config)

        writer.close()
        torch.save(self.model.state_dict(), config.experiment_name + '_final_model_checkpoint.pth')
        return episode_returns

    
    def act(self,obs,env,temp=1):
        """Use the alpha zero agent to return best action

        Args:
            obs (Union[np.array,int]): observation from the environment
            env (gym.Env): The current environment.

        Returns:
            best_action (int)
        """

        self.model.eval()
        mcts = self.mcts(env, state=obs, model=self.model, d=self.max_mcts_search_depth, m=self.num_mcts_simulations, c=self.c, gamma=self.gamma,alpha=self.alpha,epsilon=self.epsilon)
        mcts.search()
        state,pi,V,best_action= mcts.return_results(temp = temp)
        # a = np.random.choice(len(pi), p = pi) # Is this right? 
        return best_action
    



if __name__ == "__main__":
    

    ## Config contains all the hyperparameters for the model and training
    pass

    # VIN model hyperparameters

    #TODO


    # MCTS hyperparameters

    # config.gamma = 0.99
    # config.c = 1.0
    # config.num_mcts_simulations = 100
    # config.max_mcts_search_depth = 10





    # Training hyperparameters


    # Agent hyperparameters






