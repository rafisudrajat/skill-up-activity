from typing import Dict, List, Tuple
import numpy as np
import time

def check_if_movement_is_valid( 
        current_state: Tuple[int, int], 
        next_state: Tuple[int, int], 
        action: str,
        grid_height:int,
        grid_width:int 
        ) -> bool:
    """
    Checks if the movement from the current state to the next state is valid based on the action taken.

    Args:
        current_state (Tuple[int, int]): The current position in the grid as (row, column).
        next_state (Tuple[int, int]): The intended next position in the grid as (row, column).
        action (str): The action taken, one of 'up', 'down', 'left', or 'right'.
        grid_height (int): Height of the grid.
        grid_width (int): Width of the grid.

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
    if not (1 <= expected_next_state[0] <= grid_height and 1 <= expected_next_state[1] <= grid_width):
        expected_next_state = current_state  # Movement is invalid, stay in the same state

    # Return whether the actual next state matches the expected next state
    return next_state == expected_next_state

# Define transtion probabilities for each state and action pair p(s'|s,a) = 1.0 
# if the action is valid and leads to the next state (deterministic transition), otherwise 0.0
def transition_probability(
        current_state: Tuple[int, int], 
        action: str, 
        next_state: Tuple[int, int],
        grid_height:int,
        grid_width:int
        ) -> float:
    """
    Calculate the transition probability from current_state to next_state given an action.

    Args:
        current_state (Tuple[int, int]): The current position in the grid as (row, column).
        action (str): The action taken, one of 'up', 'down', 'left', or 'right'.
        next_state (Tuple[int, int]): The intended next position in the grid as (row, column).
        grid_height (int): Height of the grid.
        grid_width (int): Width of the grid.

    Returns:
        float: Transition probability.
    """
    if check_if_movement_is_valid(current_state, next_state, action, grid_height, grid_width):
        return 1.0  # Valid movement
    else:
        return 0.0  # Invalid movement
    
def reward_probability(
        current_state: Tuple[int, int],
        forbidden_states: List[Tuple[int, int]],
        goal_state: Tuple[int, int],
        grid_width: int,
        grid_height: int, 
        action: str, 
        reward: str
        ) -> float:
    """
    Calculate the probability of receiving a specific reward given the current state and action.

    Args:
        current_state (Tuple[int, int]): The current position in the grid as (row, column).
        forbidden_states (List[Tuple[int, int]]): List of forbidden states.
        goal_state (Tuple[int, int]): The goal state.
        grid_width (int): Width of the grid.
        grid_height (int): Height of the grid.
        action (str): The action taken, one of 'up', 'down', 'left', or 'right'.
        reward (str): The type of reward, one of 'forbidden', 'goal', 'boundary', or 'default'.

    Returns:
        float: Probability of receiving the specified reward (1.0 if valid, 0.0 otherwise).
    """

    forbidden_states = set(forbidden_states)  # Convert to set for faster lookup

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

    is_crossing_boundary = not (1 <= next_state[0] <= grid_height and 1 <= next_state[1] <= grid_width)
    is_forbidden_state = next_state in forbidden_states

    # Check conditions for each reward type
    if reward == 'forbidden' and is_forbidden_state:
        return 1.0
    if reward == 'goal' and next_state == goal_state:
        return 1.0
    if reward == 'boundary' and is_crossing_boundary:
        return 1.0
    if reward == 'default' and not is_crossing_boundary and not is_forbidden_state and next_state != goal_state:
        return 1.0

    return 0.0

def value_iteration(all_states: List[Tuple[int, int]],
                    value_states: Dict[Tuple[int, int], float],
                    actions: List[str],
                    rewards: Dict[str, int],
                    grid_width: int,
                    grid_height: int,
                    forbidden_states: List[Tuple[int, int]],
                    goal_state: Tuple[int, int],
                    gamma: float, 
                    max_error: float,
                    max_iter:int)-> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    """
    Perform value iteration to compute the optimal value function and policy for a given Markov Decision Process (MDP).
    Args:
        all_states (List[Tuple[int, int]]): A list of all possible states in the MDP, represented as tuples.
        value_states (Dict[Tuple[int, int], float]): A dictionary mapping each state to its current value estimate.
        actions (List[str]): A list of all possible actions that can be taken in the MDP.
        rewards (Dict[str, int]): A dictionary mapping actions to their corresponding rewards.
        grid_width (int): The width of the grid world.
        grid_height (int): The height of the grid world.
        forbidden_states (List[Tuple[int, int]]): A list of states that are forbidden (e.g., obstacles).
        goal_state (Tuple[int, int]): The goal state in the grid world.
        gamma (float): The discount factor, which determines the importance of future rewards (0 <= gamma <= 1).
        max_error (float): The maximum allowable error for convergence. The algorithm stops when the value function changes by less than this amount.
        max_iter (int): The maximum number of iterations to perform to prevent infinite loops.
    Returns:
        Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
            - A dictionary mapping each state to its optimal value.
            - A dictionary mapping each state to its optimal policy (the best action to take in that state).
    Notes:
        - The function iteratively updates the value of each state based on the Bellman equation until convergence or the maximum number of iterations is reached.
        - The policy is updated to reflect the action that maximizes the Q-value for each state.
        - Convergence is determined when the maximum change in value across all states is less than `max_error`.
    """ 
        
    def sum_of_rewards(state: Tuple[int, int], action: str) -> float:
            result = sum(reward_probability(state, forbidden_states, goal_state, grid_width, grid_height, action, reward_key) * rewards[reward_key] 
                       for reward_key in rewards)
            return result
    
    for _ in range(max_iter):

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
        transition_probability_matrix = [[transition_probability(state, action, next_state, grid_height, grid_width) for next_state in all_states] 
            for state in all_states for action in actions]

        transition_probability_matrix = np.array(transition_probability_matrix)
        
        # [R]_{s,a} \doteq \sum_{r \in \mathcal{R}} p(r|s,a) r
        # where R is the reward matrix, s is the current state, a is the action taken, and r is the reward.
        # Example for 2 states and 2 actions:
        # R[0][0] = r(s0,a0)
        # R[0][1] = r(s0,a1)
        # R[1][0] = r(s1,a0)
        # R[1][1] = r(s1,a1)
        rewards_matrix = [[sum_of_rewards(state, action) for action in actions] 
                                for state in all_states]
        
        rewards_vector = np.array(rewards_matrix).flatten()

        value_states_vector = np.array(list(value_states.values()))

        q_values_vector = rewards_vector + gamma * transition_probability_matrix @ value_states_vector
        q_values_matrix = np.array(q_values_vector).reshape(len(all_states), len(actions))

        # Policy update: choose the action with the highest Q-value for each state
        best_actions = np.argmax(q_values_matrix, axis=1)
        policies = {state: actions[best_action] for state, best_action in zip(all_states, best_actions)}
        # Value update: update the value of each state based on the best action
        value_states = {state: q_values_matrix[i, best_actions[i]] for i, state in enumerate(all_states)}
        # Check for convergence
        if np.max(np.abs(np.array(list(value_states.values())) - value_states_vector)) < max_error:
            break
    return value_states, policies


    
# Example usage
if __name__ == "__main__":
    # Assuming the environment is a grid world with 4 actions: up, down, left, right
    # Define all states and actions
    GRID_WIDTH = 5
    GRID_HEIGHT = 5
    ACTIONS = ['up', 'down', 'left', 'right', 'stay']

    # STATES represents all possible positions in a 5x5 grid world.
    # Each state is a tuple (i, j), where 'i' is the row index (0 to 4) and 'j' is the column index (0 to 4).
    # This effectively enumerates all grid cells in the environment.
    STATES = [(i, j) for i in range(1,GRID_HEIGHT+1) for j in range(1,GRID_WIDTH+1)]
    FORBIDDEN_STATES = [(2, 2), (2, 3), (3, 3), (4,2), (4,4), (5,2)]
    GOAL_STATE = (4, 3)

    VALUE_STATES = {state: 0.0 for state in STATES}  # Initialize value function for all states to 0.0
    for forbidden_state in FORBIDDEN_STATES:
        VALUE_STATES[forbidden_state] = -100  # Assign a negative value to forbidden states 
    VALUE_STATES[GOAL_STATE] = 10  # Assign a high value to the goal state

    

    REWARDS = {'boundary': -1, 'forbidden':-10, 'goal': 1, 'default': 0}  # Example rewards
    # Define the discount factor and convergence parameters
    gamma = 0.9  # Discount factor
    max_error = 10e-7  # Convergence threshold
    max_iter = 1000  # Maximum number of iterations

    # Record the start time
    start_time = time.time()
    # Perform value iteration
    optimal_values, optimal_policies = value_iteration(
        all_states=STATES,
        value_states=VALUE_STATES,
        actions=ACTIONS,
        rewards=REWARDS,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        forbidden_states=FORBIDDEN_STATES,
        goal_state=GOAL_STATE,
        gamma=gamma,
        max_error=max_error,
        max_iter=max_iter
    )
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Print the optimal values and policies
    print("Optimal Values:")
    for state, value in optimal_values.items():
        print(f"State {state}: Value = {value:.2f}")

    print("\nOptimal Policies:")
    for state, policy in optimal_policies.items():
        print(f"State {state}: Policy = {policy}")
