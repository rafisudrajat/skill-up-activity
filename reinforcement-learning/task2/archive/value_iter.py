import numpy as np
from typing import List, Tuple, Dict
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
 
# Define reward probabilities for each state and action pair p(r|s,a) = 1.0
# if the action leads to a state with the specified reward, otherwise 0.0
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

def calculate_q_value(  
        current_state: Tuple[int, int], 
        current_action: str,
        rewards: Dict[str, int], 
        value_states: Dict[Tuple[int, int], float],
        grid_width: int,
        grid_height: int,
        forbidden_states: List[Tuple[int, int]],
        goal_state: Tuple[int, int],
        gamma: float
        ) -> float:
    """
    Calculate the Q-value for a given state-action pair.

    Args:
        current_state (Tuple[int, int]): The current position in the grid as (row, column).
        current_action (str): The action taken, one of 'up', 'down', 'left', or 'right'.
        rewards (Dict[str, int]): A dictionary mapping reward types ('forbidden', 'goal', etc.) to their values.
        value_states (Dict[Tuple[int, int], float]): A dictionary mapping states to their current value estimates.
        grid_width (int): The width of the grid world.
        grid_height (int): The height of the grid world.
        forbidden_states (List[Tuple[int, int]]): A list of states that are forbidden (e.g., obstacles).
        goal_state (Tuple[int, int]): The goal state in the grid world.
        gamma (float): The discount factor for future rewards (0 <= gamma <= 1).

    Returns:
        float: The Q-value for the given state-action pair.
    """
    # Calculate the immediate reward
    immediate_reward = sum(
        reward_probability(current_state,
                           forbidden_states,
                           goal_state,
                           grid_width, 
                           grid_height , 
                           current_action, 
                           reward) * rewards[reward]
        for reward in rewards
    )

    # Calculate the expected future reward
    future_reward = sum(
        transition_probability(current_state, 
                               current_action, 
                               next_state, 
                               grid_height, 
                               grid_width) * value_states[next_state]
        for next_state in value_states
    )

    # Return the total Q-value
    return immediate_reward + gamma * future_reward

# Value Iteration Algorithm, which updates the value function and policy for all states until convergence
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
    policies = {state: None for state in all_states}  # Initialize policy for all states
    
     
    # Iterate until convergence or max_error is reached 
    for i in range(max_iter):  # Limit iterations to prevent infinite loop
        # delta = 0.0        
        # Iterate over all states to update values and policies
        for state in all_states:
            q_values = {action: calculate_q_value(state, 
                                                  action, 
                                                  rewards, 
                                                  value_states,
                                                  grid_width, 
                                                  grid_height, 
                                                  forbidden_states,
                                                  goal_state,
                                                  gamma) for action in actions}
            # Policy update: choose the action with the highest Q-value
            best_action = max(q_values, key=q_values.get)
            best_q_value = q_values[best_action]
            # Policy update
            policies[state] = best_action
            # Value update
            value_states[state] = best_q_value
            # delta = max(delta, abs(best_q_value - value_states[state]))
        # Check for convergence
        # if delta < max_error:
        #     break
        
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

    gamma = 0.9  # Discount factor
    max_error = 10e-7  # Convergence threshold
    max_iter = 1000  # Maximum number of iterations

    ####################### FOR DEBUGGING #######################
 
    # Create transition probabilities for all state-action pairs
    # This is a deterministic environment, so the transition probability is either 0 or 1.
    # The transition probability is 1 if the action leads to the next state, otherwise 0.
    # In a more complex environment, this could be a stochastic function based on the action taken.
    transition_probabilities = {
        (state, action): {next_state: transition_probability(state, action, next_state, GRID_HEIGHT, GRID_WIDTH)
                            for next_state in STATES}
        for state in STATES for action in ACTIONS
    }

    # print(transition_probabilities)  # Print transition probabilities for debugging
    # Get all possible next states for a given state and action using the value in transition_probabilities
    all_possible_next_states = []
    for state_action in transition_probabilities:
        state, action = state_action
        for next_state, prob in transition_probabilities[state_action].items():
            if prob == 1.0:  # Check if the transition probability is 1.0
                all_possible_next_states.append((state, action, next_state))

    print("All possible next states:")
    print(all_possible_next_states)
    print("Count: " ,len(all_possible_next_states))


    # Create all reward probabilities for each state and action pair
    reward_probabilities = {
        (state, action): {reward: reward_probability(state, FORBIDDEN_STATES, GOAL_STATE, GRID_WIDTH, GRID_HEIGHT, action, reward)
                            for reward in REWARDS}
        for state in STATES for action in ACTIONS
    }

    # Get all possible rewards for a given state and action using the value in reward_probabilities
    all_possible_rewards = []
    for state_action in reward_probabilities:
        state, action = state_action
        for reward, prob in reward_probabilities[state_action].items():
            if prob == 1.0:  # Check if the reward probability is 1.0
                all_possible_rewards.append((state, action, reward))
    print("All possible rewards:")
    print(all_possible_rewards)
    print("count", len(all_possible_rewards))

    ######################### FOR DEBUGGING #######################

    # Start the timer
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
    # Stop the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for value iteration: {elapsed_time:.4f} seconds")

    # Print the results
    print("Optimal Values:")
    for state, value in optimal_values.items():
        print(f"State {state}: Value {value}")

    print("\nOptimal Policies:")
    for state, policy in optimal_policies.items():
        print(f"State {state}: Policy {policy}")                
