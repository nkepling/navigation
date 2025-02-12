import unittest
import numpy as np
from new_grid_env import GridworldEnv,WrapForMCTS  # Import your environment

from copy import deepcopy


class TestGridworldEnv(unittest.TestCase):
    
    def setUp(self):
        """
        Create a 3x3 Gridworld for testing with predefined rewards and obstacles.
        """
        self.rewards = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 10]
        ])
        self.obstacles = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        
        self.env = GridworldEnv(self.rewards, self.obstacles)
        self.mcts_env = WrapForMCTS(GridworldEnv(self.rewards, self.obstacles))
        self.env.reset()

    def test_initial_state(self):
        """Ensure environment starts at (0,0) and rewards are correctly set."""
        state, _ = self.env.reset()
        self.assertTrue(np.array_equal(self.env.agent_position, np.array([0, 0])))
        self.assertTrue(np.array_equal(state["rewards"], self.rewards))

    def test_out_of_bounds(self):
        """Out of bounds moves are treated as collisions."""
        self.env.reset()
        state, reward, done, _, _ = self.env.step(0)
        agent_position = state["agent_position"]    
        self.assertTrue(np.array_equal(agent_position, np.array([0, 0])))
        self.assertTrue(not done)  # Should not terminate
        self.assertTrue(reward == -1)  # Penalty for collision


    def test_valid_move(self):
        """Test if the agent moves correctly in the grid."""
        self.env.reset()
        _, _, done, _, _ = self.env.step(2)  # Move Right
        self.assertTrue(np.array_equal(self.env.agent_position, np.array([0, 1])))
        self.assertFalse(done)  # Should not terminate

        _, reward, done, _, _ = self.env.step(1)  # Move Down
        self.assertTrue(np.array_equal(self.env.agent_position, np.array([0, 1])))  # Hit an obstacle
        self.assertTrue(not done)  # Should not terminate
        self.assertTrue(reward == -1)  # Penalty for collision

    def test_reward_collection(self):
        """Test if rewards are collected and set to 0 after stepping on a reward cell."""
        self.env.reset()
        _, reward, _, _, _ = self.env.step(2)  
        self.assertEqual(reward, 1)
        self.assertEqual(self.env.get_current_rewards()[0, 1], 0)  # Reward should be consumed

    @unittest.skip("Not implemented yet")
    def test_terminal_state(self):
        """Test if environment terminates when hitting an obstacle."""

        self.env.reset()
        self.env.step(2)  # Move Right
        _, _, done, _, _ = self.env.step(1)  # Move Down (into obstacle)
        self.assertTrue(not done)


    @unittest.skip("Not implemented yet")
    def test_mcts_transition_updates(self):
        """Ensure MCTS can properly track state transitions."""
        self.env.reset()
        
        # Move Right, should be at (0,1)
        state1, _, _, _, _ = self.env.step(2)
        self.assertTrue(np.array_equal(state1["agent_position"], np.array([0, 1])))

        # Move Down, should be at (1,1), but it's an obstacle so should terminate
        state2, _, done, _, _ = self.env.step(1)
        self.assertTrue(done)  # Ensure termination
        self.assertTrue(np.array_equal(state2["agent_position"], np.array([1, 1])))  # Position should not change

    @unittest.skip("Not implemented yet")
    def test_multiple_step_rewards(self):
        """Ensure rewards accumulate correctly when multiple steps occur."""
        self.env.reset()
        state, reward, done, _, _ = self.env.step(2)
        self.assertTrue(np.array_equal(state["agent_position"],np.array([0,1])))
        self.assertTrue(reward == 1)
        self.assertFalse(done)

        state, reward, done, _, _ = self.env.step(0)
        self.assertTrue(np.array_equal(state["agent_position"],np.array([0, 0])))
        self.assertTrue(reward == 0)

        state, reward, done, _, _ = self.env.step(2)
        self.assertTrue(np.array_equal(state["agent_position"],np.array([0, 1])))
        self.assertTrue(reward == 0)


    def test_reset(self):
        """Ensure environment resets correctly."""
        self.env.reset()
        _,reward,done,_,_ = self.env.step(2)  # Move Right
       
        
        self.env.reset()
        self.assertTrue(np.array_equal(self.env.agent_position, np.array([0, 0])))
        self.assertTrue(np.array_equal(self.env.get_current_rewards(), self.rewards))

    def test_copy(self):
        """Test the copy method.
        """

        self.env.reset()
        self.env.step(2)
        self.env.step(1)    

        env_copy = self.env.copy()

        self.assertTrue(np.array_equal(self.env.agent_position, env_copy.agent_position))
        self.assertTrue(np.array_equal(self.env.get_current_rewards(), env_copy.get_current_rewards()))
        self.assertTrue(np.array_equal(self.env.obstacles, env_copy.obstacles))

    @unittest.skip("Not implemented yet")
    def test_render(self):
        """Test the render method."""
        self.env.reset()
        self.env.render()


    def test_hash(self):
        """Test the hash method."""

        self.mcts_env.reset()

        id_1,reward,done,_,_ = self.mcts_env.step(2)

        id_2,reward,done,_,_  = self.mcts_env.step(0)

        self.assertNotEqual(id_1,id_2)

        id_3,reward,done,_,_  = self.mcts_env.step(2)

        id_6,reward,done,_,_  = self.mcts_env.step(0)

        self.assertEqual(id_2,id_6)


        copy_env = self.mcts_env.copy()



        self.assertTrue(np.array_equal(copy_env.agent_position,self.mcts_env.agent_position))
        self.assertTrue(np.array_equal(copy_env.get_current_rewards(),self.mcts_env.get_current_rewards()))
        self.assertTrue(np.array_equal(copy_env.obstacles,self.mcts_env.obstacles))
        self.assertTrue(copy_env.state_to_uid == self.mcts_env.state_to_uid)


        id_4,reward,done,_,_ = copy_env.step(1)
        id_5,reward,done,_,_  = self.mcts_env.step(1)

        self.assertEqual(id_4,id_5)

    def test_copy_hash(self):
        """Test the copy method and hash method."""

        self.mcts_env.reset()

        id_1,reward,done,_,_ = self.mcts_env.step(2)

        copy_env = self.mcts_env.copy()

        id_2,reward,done,_,_  = copy_env.step(0)
        id_3,reward,done,_,_  = self.mcts_env.step(0)

        self.assertEqual(id_2,id_3)

        copy_env2 = deepcopy(copy_env)

        copy_env3 = deepcopy(self.mcts_env)

        id_4,reward,done,_,_ = copy_env2.step(1)
        id_5,reward,done,_,_  = copy_env3.step(1)

        print("id_4",id_4)
        print("id_5",id_5)

        


        
        self.assertEqual(id_4,id_5)


    def test_get_state(self):
        """Test the get_state method."""
        self.mcts_env.reset()
        state = self.mcts_env.get_state()

        self.assertIsInstance(state,dict)


    def test_rewards(self):
        """Test the rewards method."""
        self.mcts_env.reset()
        rewards = self.mcts_env.get_current_rewards()
        self.assertTrue(np.array_equal(rewards,self.rewards))


if __name__ == '__main__':
    unittest.main()
