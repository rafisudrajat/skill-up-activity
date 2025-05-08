import numpy as np
from typing import List, Tuple, Dict

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


def policy_iteration(
        all_states: List[Tuple[int, int]],
        value_states: Dict[Tuple[int, int], float],
        policy: Dict[Tuple[int, int], str],
        actions: List[str],
        rewards: Dict[str, int],
        grid_width: int,
        grid_height: int,
        forbidden_states: List[Tuple[int, int]],
        goal_state: Tuple[int, int],
        gamma: float, 
        max_policy_eval_error: float,
        max_policy_eval_iter: int,
        max_policy_iter_error: float,
        max_policy_iter_loop: int
        )-> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    """
    Perform policy iteration to solve a reinforcement learning problem.
    Policy iteration is an iterative algorithm that alternates between policy evaluation 
    and policy improvement to find the optimal policy and value function for a given 
    Markov Decision Process (MDP).
    
    Args:
        all_states (List[Tuple[int, int]]): A list of all possible states in the grid.
        value_states (Dict[Tuple[int, int], float]): A dictionary mapping each state to its 
            current estimated value.
        policy (Dict[Tuple[int, int], str]): A dictionary mapping each state to its current 
            policy (action to take in that state).
        actions (List[str]): A list of all possible actions.
        rewards (Dict[str, int]): A dictionary mapping reward keys to their corresponding 
            reward values.
        grid_width (int): The width of the grid.
        grid_height (int): The height of the grid.
        forbidden_states (List[Tuple[int, int]]): A list of states that are not allowed 
            (e.g., obstacles).
        goal_state (Tuple[int, int]): The goal state in the grid.
        gamma (float): The discount factor for future rewards (0 <= gamma <= 1).
        max_policy_eval_error (float): The maximum allowable error for policy evaluation 
            convergence. If negative, neglect the convergence check.
        max_policy_eval_iter (int): The maximum number of iterations for policy evaluation.
        max_policy_iter_error (float): The maximum allowable error for policy iteration 
            convergence. If negative, neglect the convergence check.
        max_policy_iter_loop (int): The maximum number of iterations for policy improvement.
    
    Returns:
        Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
            - A dictionary mapping each state to its optimal value.
            - A dictionary mapping each state to its optimal policy (action to take in that state).
    """
    
    def sum_of_rewards(state: Tuple[int, int], action: str) -> float:
            return sum(reward_probability(state, forbidden_states, goal_state, grid_width, grid_height, action, reward_key) * rewards[reward_key] 
                       for reward_key in rewards)
    
    # Create a mapping of actions and states to indices for easier access
    actions_indices = {action: i for i, action in enumerate(actions)}

    states_indices = {state: i for i, state in enumerate(all_states)}
    
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
    
    # [P]_{s,a,s'} \doteq p(s'|s,a) a 3D matrix 
    # where P is the transition probability matrix, s is the current state, a is the action taken, and s' is the next state.
    transition_probability_tensor = np.array([
        [
            [transition_probability(state, action, next_state, grid_height, grid_width) 
             for action in actions]
            for next_state in all_states
        ] 
        for state in all_states
    ])

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
    # Reorder axes: (state, action, next_state) instead of (state, next_state, action)
    transition_probability_matrix = transition_probability_tensor.transpose(0, 2, 1)
    # Reshape to 2D: (num_states * num_actions, num_next_states)
    transition_probability_matrix = transition_probability_matrix.reshape(-1, transition_probability_tensor.shape[1])

    value_states_vector = np.array([value_states[state] for state in all_states])
    
    for _ in range(max_policy_iter_loop):
        # Policy evaluation
        # evaluate the current policy until convergence
        for _ in range(max_policy_eval_iter):
            r_pi = [rewards_matrix[states_indices[state]][actions_indices[policy[state]]] for state in all_states]
            r_pi = np.array(r_pi)

            P_pi = np.array([
                [
                transition_probability_tensor[states_indices[state]][states_indices[next_state]][actions_indices[policy[state]]]
                for next_state in all_states
                ]
                for state in all_states
            ])
            
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
        q_values_matrix = q_values_vector.reshape(len(all_states), len(actions))
        best_actions = np.argmax(q_values_matrix, axis=1)
        # update the policy with the best action for each state
        for i, state in enumerate(all_states):
            policy[state] = actions[best_actions[i]]
        
        # check for convergence
        delta = np.max(np.abs(np.array(list(value_states.values())) - value_states_vector))
        # if the maximum error is less than the threshold, break the loop
        if max_policy_iter_error >= 0 and delta < max_policy_iter_error:
            # update the value function
            value_states = {state: value_states_vector[i] for i, state in enumerate(all_states)}
            break
        
        # update the value function
        value_states = {state: value_states_vector[i] for i, state in enumerate(all_states)}

    return value_states, policy

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

    POLICY = {state: 'stay' for state in STATES}  # Initialize policy to 'stay' for all states

    REWARDS = {'boundary': -1, 'forbidden':-10, 'goal': 1, 'default': 0}  # Example rewards

    GAMMA = 0.9  # Discount factor
    MAX_POLICY_EVAL_ERROR = 10e-7  # Convergence threshold for policy evaluation
    MAX_POLICY_EVAL_ITER = 1000  # Maximum iterations for policy evaluation
    MAX_POLICY_ITER_ERROR = -0.5  # Convergence threshold for policy iteration
    MAX_POLICY_ITER_LOOP = 1000  # Maximum iterations for policy iteration

    # Run policy iteration
    value_states, policy = policy_iteration(
        STATES,
        VALUE_STATES,
        POLICY,
        ACTIONS,
        REWARDS,
        GRID_WIDTH,
        GRID_HEIGHT,
        FORBIDDEN_STATES,
        GOAL_STATE,
        GAMMA, 
        MAX_POLICY_EVAL_ERROR,
        MAX_POLICY_EVAL_ITER,
        MAX_POLICY_ITER_ERROR,
        MAX_POLICY_ITER_LOOP
    )

    print("Final Value States:")
    for state, value in value_states.items():
        print(f"State {state}: {value:.2f}")
    
    print("\nFinal Policy:")
    for state, action in policy.items():
        print(f"State {state}: {action}")
