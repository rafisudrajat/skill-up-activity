try:
    from .defined_types import Action, Policy, ValueFunction, State
    from .grid_world_env import GridWorldEnvironment
except ImportError:
    from defined_types import Action, Policy, ValueFunction, State
    from grid_world_env import GridWorldEnvironment

import numpy as np

class PolicyIteration():
    """
    A class that implements the Policy Iteration algorithm to find the optimal policy
    for a given GridWorld environment.

    Policy Iteration consists of two main steps:
    1. Policy Evaluation: Calculate the value function for the current policy.
    2. Policy Improvement: Update the policy to be greedy with respect to the current value function.
    These steps are repeated until the policy converges (stabilizes).
    """
    def __init__(self, env:GridWorldEnvironment, initial_policy: Policy, initial_value_function:ValueFunction):
        """
        Initialize the PolicyIteration algorithm.

        Args:
            env (GridWorldEnvironment): The grid world environment.
            initial_policy (Policy): The starting policy mapping states to actions.
            initial_value_function (ValueFunction): The starting value function mapping states to values.
        """
        self.env = env
        self.policy = initial_policy
        self.value_function = initial_value_function

    def _calculate_q_value(self, state:State, action:Action, gamma:float) -> float:
        """Calculates the Q-value (expected return) for a specific state-action pair."""
        next_state, reward = self.env.step(state, action)
        return reward + gamma * self.value_function[next_state]

    def policy_evaluation(self, gamma=0.9, theta=1e-6):
        """
        Evaluate the current policy by iteratively updating the value function until convergence.

        Args:
            gamma (float): The discount factor (0 <= gamma <= 1).
            theta (float): The threshold for convergence.
        """
        while True:
            delta = 0.0
            for row in range(self.env.grid_size.rows):
                for col in range(self.env.grid_size.cols):
                    state = (row, col)
                    action = self.policy[state]
                    
                    new_value = self._calculate_q_value(state, action, gamma)
                    
                    delta = max(delta, abs(self.value_function[state] - new_value))
                    self.value_function[state] = new_value
            
            if delta < theta:
                break
    
    def find_best_policy(self, gamma=0.9, theta=1e-6)-> Policy:
        """
        Find the optimal policy by alternating between policy evaluation and policy improvement.

        Args:
            gamma (float): The discount factor.
            theta (float): The convergence threshold for policy evaluation.

        Returns:
            Policy: The optimal policy mapping states to actions.
        """
        actions = list(Action)
        while True:
            self.policy_evaluation(gamma, theta)
            policy_stable = True
            
            # Policy update/improvement
            for row in range(self.env.grid_size.rows):
                for col in range(self.env.grid_size.cols):
                    state = (row, col)
                    old_action = self.policy[state]
                    
                    # Find the action that maximizes the Q-value
                    best_action = max(actions, key=lambda a: self._calculate_q_value(state, a, gamma))
                    
                    self.policy[state] = best_action
                    
                    if old_action != best_action:
                        policy_stable = False
            
            if policy_stable:
                break
        return self.policy
    
class ValueIteration():
    """
    A class that implements the Value Iteration algorithm to find the optimal policy
    for a given GridWorld environment.

    Value Iteration works by iteratively updating the value function using the 
    Bellman Optimality Equation. Once the value function converges, the optimal 
    policy is derived by selecting the action that maximizes the expected return 
    for each state.
    """
    def __init__(self, env:GridWorldEnvironment, initial_value_function:ValueFunction):
        """
        Initialize the ValueIteration algorithm.

        Args:
            env (GridWorldEnvironment): The grid world environment.
            initial_value_function (ValueFunction): The starting value function mapping states to values.
        """
        self.env = env
        self.value_function = initial_value_function

    def _calculate_q_values(self, state:State, gamma:float) -> tuple[list[float], list[Action]]:
        """
        Calculate Q-values (expected returns) for all possible actions in a given state.

        Args:
            state (State): The current state.
            gamma (float): The discount factor.

        Returns:
            tuple[list[float], list[Action]]: A tuple containing a list of Q-values 
            and a corresponding list of actions.
        """
        actions = list(Action)
        q_values = []
        for action in actions:
            next_state, reward = self.env.step(state, action)
            q_values.append(reward + gamma * self.value_function[next_state])
        return q_values, actions


    def find_best_policy(self, gamma=0.9, theta=1e-6)-> Policy:
        """
        Find the optimal policy by iterating on the value function until convergence.

        Args:
            gamma (float): The discount factor (0 <= gamma <= 1).
            theta (float): The convergence threshold for the value function.

        Returns:
            Policy: The optimal policy mapping states to actions.
        """
        # Calculate optimal value function
        while True:
            delta = 0.0
            for row in range(self.env.grid_size.rows):
                for col in range(self.env.grid_size.cols):
                    state = (row, col)
                    q_values, _ = self._calculate_q_values(state, gamma)
                    
                    max_value = max(q_values)
                    delta = max(delta, abs(self.value_function[state] - max_value))
                    self.value_function[state] = max_value
            
            if delta < theta:
                break
        
        # Find the best policy using optimal value function
        policy: Policy = {}
        for row in range(self.env.grid_size.rows):
            for col in range(self.env.grid_size.cols):
                state = (row, col)
                q_values, actions = self._calculate_q_values(state, gamma)
                
                best_action = actions[np.argmax(q_values)]
                policy[state] = best_action
        return policy