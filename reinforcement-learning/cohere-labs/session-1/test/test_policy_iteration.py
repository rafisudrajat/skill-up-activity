import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Set

from defined_types import GridSize, State, Policy,ValueFunction, Action
from grid_world_env import GridWorldEnvironment
from algorithm import PolicyIteration

OPTIMAL_POLICY: Policy = {
        (0,0): Action.right,
        (1,0): Action.right,
        (2,0): Action.right,
        (3,0): Action.down,
        (4,0): Action.down,
        (0,1): Action.up,
        (1,1): Action.up,
        (2,1): Action.right,
        (3,1): Action.down,
        (4,1): Action.down,
        (0,2): Action.up,
        (1,2): Action.left,
        (2,2): Action.down,
        (3,2): Action.right,
        (4,2): Action.down,
        (0,3): Action.up,
        (1,3): Action.right,
        (2,3): Action.none,
        (3,3): Action.left,
        (4,3): Action.down, 
        (0,4): Action.up,
        (1,4): Action.right,
        (2,4): Action.up,
        (3,4): Action.left,
        (4,4): Action.left,
    }

OPTIMAL_VALUE: ValueFunction = {
        (0,0):3.5,
        (1,0):3.9,
        (2,0):4.3,
        (3,0):4.8,
        (4,0):5.3,
        (0,1):3.1,
        (1,1):3.5,
        (2,1):4.8,
        (3,1):5.3,
        (4,1):5.9,
        (0,2):2.8,
        (1,2):2.5,
        (2,2):10.0,
        (3,2):5.9,
        (4,2):6.6,
        (0,3):2.5,
        (1,3):10.0,
        (2,3):10.0,
        (3,3):10.0,
        (4,3):7.3, 
        (0,4):2.3,
        (1,4):9.0,
        (2,4):10.0,
        (3,4):9.0,
        (4,4):8.1,
    }

INITIAL_POLICY: Policy = {
        (0,0):Action.right,
        (1,0):Action.right,
        (2,0):Action.right,
        (3,0):Action.right,
        (4,0):Action.right,
        (0,1):Action.right,
        (1,1):Action.right,
        (2,1):Action.right,
        (3,1):Action.right,
        (4,1):Action.right,
        (0,2):Action.right,
        (1,2):Action.right,
        (2,2):Action.right,
        (3,2):Action.right,
        (4,2):Action.right,
        (0,3):Action.right,
        (1,3):Action.right,
        (2,3):Action.right,
        (3,3):Action.right,
        (4,3):Action.right, 
        (0,4):Action.right,
        (1,4):Action.right,
        (2,4):Action.right,
        (3,4):Action.right,
        (4,4):Action.right,
    }

INITIAL_VALUE: ValueFunction = {
        (0,0):0.0,
        (1,0):0.0,
        (2,0):0.0,
        (3,0):0.0,
        (4,0):0.0,
        (0,1):0.0,
        (1,1):0.0,
        (2,1):0.0,
        (3,1):0.0,
        (4,1):0.0,
        (0,2):0.0,
        (1,2):0.0,
        (2,2):0.0,
        (3,2):0.0,
        (4,2):0.0,
        (0,3):0.0,
        (1,3):0.0,
        (2,3):0.0,
        (3,3):0.0,
        (4,3):0.0, 
        (0,4):0.0,
        (1,4):0.0,
        (2,4):0.0,
        (3,4):0.0,
        (4,4):0.0,
    }

class TestPolicyIteration(unittest.TestCase):
    def setUp(self):
        self.grid_size = GridSize(rows=5,cols=5)
        self.forbidden_states:Set[State] = {(1,1),(2,1),(2,2),(1,3),(3,3),(1,4)}
        self.goal_state:State = (2,3)
        self.env = GridWorldEnvironment(grid_size=self.grid_size,
                                        forbidden_states=self.forbidden_states,
                                        goal_state=self.goal_state)
    
    def test_policy_evaluation(self):
        algorithm = PolicyIteration(self.env, OPTIMAL_POLICY.copy(), INITIAL_VALUE.copy())

        algorithm.policy_evaluation()

        for state, value in OPTIMAL_VALUE.items():
            self.assertAlmostEqual(algorithm.value_function[state], value, places=1)

    def test_find_best_policy(self):
        algorithm = PolicyIteration(self.env, INITIAL_POLICY.copy(), INITIAL_VALUE.copy())
        best_policy = algorithm.find_best_policy()

        # Test if the optimal policy calculated from policy iteration has the same state value as the known optimal policy 
        # Optimal policy is not unique, but the optimal state value is unique, so we compare value functions
        algorithm_best_policy = PolicyIteration(self.env, best_policy.copy(), INITIAL_VALUE.copy())
        algorithm_best_policy.policy_evaluation()
        
        for state, value in OPTIMAL_VALUE.items():
            self.assertAlmostEqual(algorithm_best_policy.value_function[state], value, places=1)
        
if __name__ == '__main__':
    unittest.main()
