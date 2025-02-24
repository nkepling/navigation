import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mcts import MCTS
from tqdm import tqdm
from .experiment_setup import read_config,run_experiment

def new_mcts_agent(env,state,config,seed=None):
        return MCTS(env=env,
                    state=state,
                    d=config["d"],
                    m=config["m"],
                    c=config["c"],
                    gamma=config["gamma"],
                    heuristic=config["heuristic"],
                    vin=config["vin"],
                    puct=config["puct"],
                    temperature=config["temperature"],
                    seed=seed
                    )

def create_agent():
     return new_mcts_agent


def main(config_path):
    print("Running baseline MCTS experiment.")
    config  = read_config(config_path)
    agent_factory = create_agent()
    run_experiment(config,agent_factory,static=True)
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str,required=True)

    args = parser.parse_args()
    main(args.config_path)
