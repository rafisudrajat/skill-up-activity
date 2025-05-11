from MDP import *
    
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
    GOAL_STATES = [(4, 3)]

    VALUE_STATES = {state: 0.0 for state in STATES}  # Initialize value function for all states to 0.0
    for forbidden_state in FORBIDDEN_STATES:
        VALUE_STATES[forbidden_state] = -100  # Assign a negative value to forbidden states 
    
    for goal_state in GOAL_STATES:
        # Assign a high value to the goal state
        VALUE_STATES[goal_state] = 10
    

    REWARDS = {'boundary': -1, 'forbidden':-10, 'goal': 1, 'default': 0}  # Example rewards
    # Define the discount factor and convergence parameters
    gamma = 0.9  # Discount factor
    max_error = 10e-7  # Convergence threshold
    max_iter = 1000  # Maximum number of iterations

    # Create an instance of the GridWorldMDP class
    grid_world = GridWorldMDP(
        grid_width=GRID_WIDTH, 
        grid_height=GRID_HEIGHT,
        goal_states=GOAL_STATES,
        forbidden_states=FORBIDDEN_STATES,
        actions=ACTIONS,
        rewards=REWARDS
    )

    # Perform value iteration
    optimal_values, optimal_policies = grid_world.value_iteration(
        value_states=VALUE_STATES,
        gamma=gamma,
        max_error=max_error,
        max_iter=max_iter
    )

    # Print the optimal values and policies
    print("Optimal Values:")
    for state, value in optimal_values.items():
        print(f"State {state}: Value = {value:.2f}")

    print("\nOptimal Policies:")
    for state, policy in optimal_policies.items():
        print(f"State {state}: Policy = {policy}")
