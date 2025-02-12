from mcts import MCTS
from pytorch_value_iteration_networks.model import *
import torch
from nn_training import reformat_input

"""
Assume car is obsevable but we do not know where it is going to turn? 
"""


class VINHeuristic:
    def __init__(self,vin):
        self.vin = vin
        self.vin.eval()

    def heuristic(self,state):
        assert isinstance(state, dict)
        input = reformat_input(state["rewards"],state["obstacles"])
        _,_,value = self.vin(input)
        x = state["agent_position"][0]
        y = state["agent_position"][1]        

        return value[:, 0, x, y] 


if __name__ == "__main__":
    import argparse
    from mcts import MCTS
     
    parser = argparse.ArgumentParser()
          
    # VIN-specific parameters
    parser.add_argument('--k', type=int, default=50, help='Number of Value Iterations')
    parser.add_argument('--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_sz', type=int, default=1, help='Batch size')

    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vin_weights = torch.load('/Users/nathankeplinger/Documents/Vanderbilt/Research/fullyObservableNavigation/pytorch_value_iteration_networks/trained/vin_20x20_k_50.pth', weights_only=True, map_location=device)

    vin = VIN(config)

    vin.load_state_dict(vin_weights)

    vin.to(device)
    vin.eval()


    














