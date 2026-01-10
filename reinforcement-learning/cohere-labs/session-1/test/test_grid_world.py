import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defined_types import GridSize, Action
from grid_world_env import GridWorldEnvironment
 

class TestGridWorldEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a 5x5 grid
        self.grid_size = GridSize(rows=5, cols=5)
        self.forbidden_states = {(2, 2), (3, 3)}
        self.goal_state = (4, 4)
        self.env = GridWorldEnvironment(self.grid_size, self.forbidden_states, self.goal_state)

    def test_initialization(self):
        self.assertEqual(self.env.grid_size, self.grid_size)
        self.assertEqual(self.env.forbidden_states, self.forbidden_states)
        self.assertEqual(self.env.goal_state, self.goal_state)

    def test_step_valid_move_down(self):
        # Start at (0, 0), move down to (1, 0)
        state = (0, 0)
        action = Action.down
        next_state, reward = self.env.step(state, action)
        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, 0)

    def test_step_valid_move_right(self):
        # Start at (0, 0), move right to (0, 1)
        state = (0, 0)
        action = Action.right
        next_state, reward = self.env.step(state, action)
        self.assertEqual(next_state, (1, 0))
        self.assertEqual(reward, 0)

    def test_step_boundary_up(self):
        # Start at (0, 0), move up (invalid) -> stay at (0, 0)
        state = (0, 0)
        action = Action.up
        next_state, reward = self.env.step(state, action)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, -1)

    def test_step_boundary_left(self):
        # Start at (0, 0), move left (invalid) -> stay at (0, 0)
        state = (0, 0)
        action = Action.left
        next_state, reward = self.env.step(state, action)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, -1)

    def test_step_forbidden_state(self):
        # Start at (1, 2), move down to (2, 2) (forbidden)
        state = (2, 1)
        action = Action.down
        next_state, reward = self.env.step(state, action)
        # The step function returns the next state even if it is forbidden, 
        # but gives a negative reward.
        self.assertEqual(next_state, (2, 2))
        self.assertEqual(reward, -10)

    def test_step_reach_goal(self):
        # Start at (3, 4), move down to (4, 4) (goal)
        state = (4, 3)
        action = Action.down
        next_state, reward = self.env.step(state, action)
        self.assertEqual(next_state, (4, 4))
        self.assertEqual(reward, 1)

if __name__ == '__main__':
    unittest.main()
