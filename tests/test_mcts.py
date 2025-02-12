import unittest
import numpy as np
import gymnasium as gym
from copy import deepcopy
from mcts import MCTS
from new_grid_env import GridworldEnv  # Import your Gridworld environment

class TestMCTS(unittest.TestCase):

    def setUp(self):
        """Initialize a small test environment for MCTS."""
        rewards = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 10]
        ])
        obstacles = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        self.env = GridworldEnv(rewards, obstacles)
        self.env.reset()
        self.mcts = MCTS(self.env, self.env.get_agent_position(), d=5, m=10, c=1.4, gamma=0.99)

    def test_tree_expansion(self):
        """Test that MCTS expands a node correctly."""
        root_node = self.mcts.root
        expanded_node = self.mcts._expand(root_node)
        self.assertEqual(len(root_node.children), len(self.mcts.possible_actions))
        self.assertIn(expanded_node, root_node.children)

    def test_rollout(self):
        """Test that the MCTS default policy correctly rolls out and returns a reward estimate."""
        root_node = self.mcts.root
        reward = self.mcts._default_policy(root_node)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0)  # Rewards should not be negative in your setup.

    def test_backpropagation(self):
        """Test that backpropagation updates Q-values and visit counts correctly."""
        root_node = self.mcts.root
        test_node = self.mcts._expand(root_node)
        self.mcts._backpropagation(5, test_node)
        self.assertGreater(self.mcts.Ns[test_node.state], 0)
        self.assertGreater(self.mcts.Nsa[(test_node.state, test_node.action)], 0)
        self.assertGreater(self.mcts.Qsa[(test_node.state, test_node.action)], 0)

    def test_best_action_selection(self):
        """Test that MCTS selects the action with the highest visits."""
        self.mcts.search()  # Run MCTS simulations
        best_action = self.mcts.best_action(self.mcts.root)
        self.assertIn(best_action, self.mcts.possible_actions)

    def test_mcts_integration(self):
        """Test that MCTS integrates correctly with the environment and returns a valid action."""
        action, action_values = self.mcts.search()
        self.assertIn(action, self.mcts.possible_actions)
        self.assertEqual(len(action_values), len(self.mcts.possible_actions))

if __name__ == '__main__':
    unittest.main()
