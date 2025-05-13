import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GridWorldMDP:
    """
    A class to represent a grid world environment for policy iteration and value iteration in Markov Decision Process. 
    This class provides methods to define the grid world, visualize it, and perform policy iteration or value iteration
    to find the optimal policy and value function.

    Attributes:
        grid_width (int): The width of the grid world.
        grid_height (int): The height of the grid world.
        all_states (List[Tuple[int, int]]): A list of all possible states in the grid world, 
            represented as (row, column) tuples.
        goal_states (List[Tuple[int, int]]): A list of goal states in the grid world.
        forbidden_states (List[Tuple[int, int]]): A list of forbidden states in the grid world.
        actions (List[str]): A list of possible actions in the grid world (e.g., ["up", "down", "left", "right"]).
        rewards (Dict[str, int]): A dictionary mapping specific actions or states to their corresponding rewards.
    
    Methods:
        __init__(grid_width, grid_height, goal_states, forbidden_states, actions, rewards):
            Initializes the grid world environment with the specified dimensions, states, actions, and rewards.
        plot_grid_world():
            Draws a visual representation of the grid world, highlighting forbidden and goal states.
        policy_iteration(value_states, policy, gamma, max_policy_eval_error, max_policy_eval_iter, 
                         max_policy_iter_error, max_policy_iter_loop):
            Performs policy iteration to find the optimal policy and value function for the grid world.
        value_iteration(value_states, gamma, max_error, max_iter):
            Performs value iteration to compute the optimal value function and policy for the grid world.
    """

    def __init__(
            self, 
            grid_width: int, 
            grid_height: int,
            goal_states: List[Tuple[int, int]],
            forbidden_states: List[Tuple[int, int]],
            actions: List[str],
            rewards: Dict[str, int]
            )-> None:
        """
        Initializes the grid world environment.

        Args:
            grid_width: The width of the grid world.
            grid_height: The height of the grid world.
            goal_states: A list of goal states, where each state is represented as a tuple (row, column).
            forbidden_states: A list of forbidden states, where each state is represented as a tuple (row, column).
            actions: A list of possible actions that can be taken in the grid world (e.g., ["up", "down", "left", "right"]).
            rewards: A dictionary mapping specific actions or states to their corresponding rewards.

        Returns:
            None
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        # Define all possible states in the grid world
        # Each state is represented as a tuple (row, column)
        # The grid is 1-indexed, so we start from (1, 1) to (grid_height, grid_width)
        self.all_states = [(i, j) for i in range(1, grid_height + 1) for j in range(1, grid_width + 1)]
        self.goal_states = goal_states
        self.forbidden_states = forbidden_states
        self.actions = actions
        self.rewards = rewards

    def plot_grid_world(self) -> None:
        """
        Draws a grid world with specified forbidden (red) and goal (blue) states.
        """
        fig, ax = plt.subplots(figsize=(self.grid_width, self.grid_height))
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes
        plt.title("Grid World")

        # Legend for forbidden and goal states
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='none', label='Forbidden State'),
            Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='none', label='Goal State')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # Draw forbidden states (red)
        for row, col in self.forbidden_states:
            x_min = col - 1  # Convert to 0-indexed x
            y_min = self.grid_height - row  # Convert to 0-indexed y (origin at bottom)
            rect = Rectangle((x_min, y_min), 1, 1, facecolor='red', edgecolor='none')
            ax.add_patch(rect)

        # Draw goal states (blue)
        for row, col in self.goal_states:
            x_min = col - 1
            y_min = self.grid_height - row
            rect = Rectangle((x_min, y_min), 1, 1, facecolor='blue', edgecolor='none')
            ax.add_patch(rect)

        # Draw grid lines (on top of colored cells)
        for x in range(self.grid_width + 1):
            ax.axvline(x, color='black', lw=2)
        for y in range(self.grid_height + 1):
            ax.axhline(y, color='black', lw=2)

        plt.show()

    def policy_iteration(
            self,
            value_states: Dict[Tuple[int, int], float],
            policy: Dict[Tuple[int, int], str],
            gamma: float, 
            max_policy_eval_error: float,
            max_policy_eval_loop: int,
            max_policy_iter_error: float,
            max_policy_iter_loop: int
            )-> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
        """
        Perform policy iteration to solve a reinforcement learning problem.
        Policy iteration is an iterative algorithm that alternates between policy evaluation 
        and policy improvement to find the optimal policy and value function for a given 
        Markov Decision Process (MDP).
        
        Args:
            value_states: A dictionary mapping each state to its 
                current estimated initial value.
            policy: A dictionary mapping each state to its current 
                policy (action to take in that state).
            gamma: The discount factor for future rewards (0 <= gamma <= 1).
            max_policy_eval_error: The maximum allowable error for policy evaluation 
                convergence. If negative, neglect the convergence check.
            max_policy_eval_loop: The maximum number of iterations for policy evaluation.
            max_policy_iter_error: The maximum allowable error for policy iteration 
                convergence. If negative, neglect the convergence check.
            max_policy_iter_loop: The maximum number of iterations for policy improvement.
        
        Returns:
            Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
                - A dictionary mapping each state to its optimal value.
                - A dictionary mapping each state to its optimal policy (action to take in that state).
        """
        
        # Create a mapping of actions to indices for easier access
        actions_indices = {action: i for i, action in enumerate(self.actions)}
        
        # [R]_{s,a} \doteq \sum_{r \in \mathcal{R}} p(r|s,a) r
        # where R is the reward matrix, s is the current state, a is the action taken, and r is the reward.
        # Example for 2 states and 2 actions:
        # R[0][0] = r(s0,a0)
        # R[0][1] = r(s0,a1)
        # R[1][0] = r(s1,a0)
        # R[1][1] = r(s1,a1)
        rewards_matrix = np.array([[self.__sum_of_rewards(state, action) for action in self.actions] 
                                for state in self.all_states])
        
        rewards_vector = rewards_matrix.flatten()
        
        # [P]_{s,a,s'} \doteq p(s'|s,a) a 3D matrix 
        # where P is the transition probability matrix, s is the current state, a is the action taken, and s' is the next state.
        #  (3D: state, action, next_state)
        transition_probability_tensor = np.array([
            [self.__transition_probability(state, action, next_state)
                for next_state in self.all_states]
            for state in self.all_states
            for action in self.actions
        ]).reshape(len(self.all_states), len(self.actions), len(self.all_states))

        # [P]_{(s,a),s'} \doteq p(s'|s,a) a 2D matrix
        # where P is the transition probability matrix, (s,a) is the current state-action pair, and s' is the next state.
        # Example for 2 states and 2 actions:
        # P[0][0] = p(s0|s0,a0)
        # P[0][1] = p(s1|s0,a0)
        # P[1][0] = p(s0|s0,a1)
        # P[1][1] = p(s1|s0,a1)
        # P[2][0] = p(s0|s1,a0)
        # P[2][1] = p(s1|s1,a0)
        # P[3][0] = p(s0|s1,a1)
        # P[3][1] = p(s1|s1,a1)
        # Reshape the tensor to a 2D matrix for easier access (2D: (state*action) x next_state)
        transition_probability_matrix = transition_probability_tensor.reshape(-1, len(self.all_states))

        value_states_vector = np.array([value_states[state] for state in self.all_states])
        
        for _ in range(max_policy_iter_loop):
            # Get policy actions for all states (vectorized)
            policy_actions = np.array([actions_indices[policy[state]] for state in self.all_states])

            # Get transition probabilities for the current policy (vectorized)
            P_pi = transition_probability_tensor[np.arange(len(self.all_states)), policy_actions, :]

            # Get rewards for the current policy (vectorized)
            r_pi = rewards_matrix[np.arange(len(self.all_states)), policy_actions]

            # Policy evaluation
            # evaluate the current policy until convergence
            for _ in range(max_policy_eval_loop):
                # calculate the new value function using the Bellman equation
                new_value_states_vector = r_pi + gamma * P_pi @ value_states_vector
                # check for convergence
                delta = np.max(np.abs(new_value_states_vector - value_states_vector))
                # if the maximum error is less than the threshold, break the loop
                
                if max_policy_eval_error >= 0 and delta < max_policy_eval_error:
                    value_states_vector = new_value_states_vector
                    break
                value_states_vector = new_value_states_vector
            
            # Policy improvement
            # update the policy based on the new value function
            q_values_vector = rewards_vector + gamma * transition_probability_matrix @ value_states_vector
            q_values_matrix = q_values_vector.reshape(len(self.all_states), len(self.actions))
            best_actions = np.argmax(q_values_matrix, axis=1)
            # update the policy with the best action for each state
            for i, state in enumerate(self.all_states):
                policy[state] = self.actions[best_actions[i]]
            
            # check for convergence
            delta = np.max(np.abs(np.array(list(value_states.values())) - value_states_vector))
            # if the maximum error is less than the threshold, break the loop
            if max_policy_iter_error >= 0 and delta < max_policy_iter_error:
                # update the value function
                value_states = {state: value_states_vector[i] for i, state in enumerate(self.all_states)}
                break
            
            # update the value function
            value_states = {state: value_states_vector[i] for i, state in enumerate(self.all_states)}

        return value_states, policy
    
    def value_iteration(
            self,
            value_states: Dict[Tuple[int, int], float],
            gamma: float, 
            max_error: float,
            max_iter:int
            )-> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
        """
        Perform value iteration to compute the optimal value function and policy for a given Markov Decision Process (MDP).
        Args:
            value_states: A dictionary mapping each state to its current value estimate.
            gamma: The discount factor, which determines the importance of future rewards (0 <= gamma <= 1).
            max_error: The maximum allowable error for convergence. The algorithm stops when the value function changes by less than this amount.
            max_iter: The maximum number of iterations to perform to prevent infinite loops.
        Returns:
            Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
                - A dictionary mapping each state to its optimal value.
                - A dictionary mapping each state to its optimal policy (the best action to take in that state).
        Notes:
            - The function iteratively updates the value of each state based on the Bellman equation until convergence or the maximum number of iterations is reached.
            - The policy is updated to reflect the action that maximizes the Q-value for each state.
            - Convergence is determined when the maximum change in value across all states is less than `max_error`.
        """ 
        
        # [P]_{(s,a),s^{\prime}} \doteq p(s^{\prime}|s,a)
        # where P is the transition probability matrix, s is the current state, a is the action taken, and s' is the next state.
        # Example for 2 states and 2 actions:
        # P[0][0] = p(s0|s0,a0)
        # P[0][1] = p(s1|s0,a0)
        # P[1][0] = p(s0|s0,a1)
        # P[1][1] = p(s1|s0,a1)
        # P[2][0] = p(s0|s1,a0)
        # P[2][1] = p(s1|s1,a0)
        # P[3][0] = p(s0|s1,a1)
        # P[3][1] = p(s1|s1,a1)
        transition_probability_matrix = np.array([[self.__transition_probability(state, action, next_state) for next_state in self.all_states] 
            for state in self.all_states for action in self.actions])
        
        # [R]_{s,a} \doteq \sum_{r \in \mathcal{R}} p(r|s,a) r
        # where R is the reward matrix, s is the current state, a is the action taken, and r is the reward.
        # Example for 2 states and 2 actions:
        # R[0][0] = r(s0,a0)
        # R[0][1] = r(s0,a1)
        # R[1][0] = r(s1,a0)
        # R[1][1] = r(s1,a1)
        rewards_matrix = np.array([[self.__sum_of_rewards(state, action) for action in self.actions] 
                                for state in self.all_states])
        
        rewards_vector = rewards_matrix.flatten()

        for _ in range(max_iter):
            value_states_vector = np.array(list(value_states.values()))

            q_values_vector = rewards_vector + gamma * transition_probability_matrix @ value_states_vector
            q_values_matrix = q_values_vector.reshape(len(self.all_states), len(self.actions))
            
            # Policy update: choose the action with the highest Q-value for each state
            best_actions = np.argmax(q_values_matrix, axis=1)
            policies = {state: self.actions[best_action] for state, best_action in zip(self.all_states, best_actions)}
            # Value update: update the value of each state based on the best action
            value_states = {state: q_values_matrix[i, best_actions[i]] for i, state in enumerate(self.all_states)}
            # Check for convergence
            if np.max(np.abs(np.array(list(value_states.values())) - value_states_vector)) < max_error:
                break
        return value_states, policies
    
    def __check_if_movement_is_valid(
        self,
        current_state: Tuple[int, int], 
        next_state: Tuple[int, int], 
        action: str, 
        ) -> bool:
        """
        Checks if the movement from the current state to the next state is valid based on the action taken.

        Args:
            current_state: The current position in the grid as (row, column).
            next_state: The intended next position in the grid as (row, column).
            action: The action taken, one of 'up', 'down', 'left', or 'right'.

        Returns:
            bool: True if the movement is valid, False otherwise.
        """
        # Define the expected movement for each action
        action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'stay': (0, 0)
        }

        # Calculate the expected next state
        expected_next_state = (
            current_state[0] + action_deltas[action][0],
            current_state[1] + action_deltas[action][1]
        )

        # Check for boundary conditions
        if not (1 <= expected_next_state[0] <= self.grid_height and 1 <= expected_next_state[1] <= self.grid_width):
            expected_next_state = current_state  # Movement is invalid, stay in the same state

        # Return whether the actual next state matches the expected next state
        return next_state == expected_next_state

    # Define transtion probabilities for each state and action pair p(s'|s,a) = 1.0 
    # if the action is valid and leads to the next state (deterministic transition), otherwise 0.0
    def __transition_probability(
            self,
            current_state: Tuple[int, int], 
            action: str, 
            next_state: Tuple[int, int]
            ) -> float:
        """
        Calculate the transition probability from current_state to next_state given an action.

        Args:
            current_state: The current position in the grid as (row, column).
            action: The action taken, one of 'up', 'down', 'left', or 'right'.
            next_state: The intended next position in the grid as (row, column).

        Returns:
            float: Transition probability.
        """
        if self.__check_if_movement_is_valid(current_state, next_state, action):
            return 1.0  # Valid movement
        else:
            return 0.0  # Invalid movement
    
    def __reward_probability(
            self,
            current_state: Tuple[int, int],
            action: str, 
            reward: str
            ) -> float:
        """
        Calculate the probability of receiving a specific reward given the current state and action.

        Args:
            current_state: The current position in the grid as (row, column).
            action: The action taken, one of 'up', 'down', 'left', or 'right'.
            reward: The type of reward, one of 'forbidden', 'goal', 'boundary', or 'default'.

        Returns:
            float: Probability of receiving the specified reward (1.0 if valid, 0.0 otherwise).
        """

        forbidden_states = set(self.forbidden_states)  # Convert to set for faster lookup
        goal_states = set(self.goal_states)  # Convert to set for faster lookup

        # Define the expected movement for each action
        action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'stay': (0, 0)
        }

        # Calculate the expected next state
        next_state = (
            current_state[0] + action_deltas[action][0],
            current_state[1] + action_deltas[action][1]
        )

        is_crossing_boundary = not (1 <= next_state[0] <= self.grid_height and 1 <= next_state[1] <= self.grid_width)
        is_forbidden_state = next_state in forbidden_states

        # Check conditions for each reward type
        if reward == 'forbidden' and is_forbidden_state:
            return 1.0
        if reward == 'goal' and next_state in goal_states:
            return 1.0
        if reward == 'boundary' and is_crossing_boundary:
            return 1.0
        if reward == 'default' and not is_crossing_boundary and not is_forbidden_state and next_state not in goal_states:
            return 1.0

        return 0.0

    def __sum_of_rewards(self, state: Tuple[int, int], action: str) -> float:
        '''
        Calculate the sum of rewards weighted by their probabilities for a given state and action.
        
        Args:
            state: The current position in the grid as (row, column).
            action: The action taken, one of 'up', 'down', 'left', or 'right'.
        
        Returns:
            float: The sum of rewards weighted by their probabilities.
        '''     
        return sum(self.__reward_probability(state, action, reward_key) * self.rewards[reward_key] 
            for reward_key in self.rewards)
    