import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stl_mcts import STLMCTS,grab_area_of_interest
from invariant_functions import *
from tqdm import tqdm
from .experiment_setup import read_config,run_experiment,run_static_episode
from pytorch_value_iteration_networks.model import * 
from types import SimpleNamespace
import torch



def new_mcts_agent(env,state,config,seed=None):
        
        vin = VIN(SimpleNamespace(**config))
        vin_weights = torch.load(config["vin_model_weights"],weights_only=True, map_location=config["device"])

        vin.load_state_dict(vin_weights)

        reward_map = env.unwrapped.starting_rewards

        spec1 = grab_area_of_interest(reward_map)

        spec_list = [spec1]

        return STLMCTS(env=env,
                    state=state,
                    d=config["d"],
                    m=config["m"],
                    c=config["c"],
                    gamma=config["gamma"],
                    heuristic=config["heuristic"],
                    c_heuristic=config["c_heuristic"],
                    vin=vin,
                    puct=config["puct"],
                    temperature=config["temperature"],
                    seed=seed,
                    device=config["device"],
                    k=config["k"],
                    rho_min=config["rho_min"],
                    rho_high=config["rho_high"],
                    alpha_increase=config["alpha_increase"],
                    alpha_decrease=config["alpha_decrease"],
                    check_frequency=config["check_frequency"],
                    spec_function_list=spec_list
                        )

def create_agent():
     return new_mcts_agent


def main(config_path,vin_model_weights):
    print("Running baseline MCTS experiment.")
    config  = read_config(config_path)
    config["vin_model_weights"] = vin_model_weights
    agent_factory = create_agent()
    run_experiment(config,agent_factory,static=False)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str,required=True)
    parser.add_argument('--vin_model_weights',type=str,required=True)

    args = parser.parse_args()
    main(args.config_path,args.vin_model_weights)
