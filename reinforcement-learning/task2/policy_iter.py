from MDP import *

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
    
    POLICY = {state: 'stay' for state in STATES}  # Initialize policy to 'stay' for all states

    REWARDS = {'boundary': -1, 'forbidden':-10, 'goal': 1, 'default': 0}  # Example rewards

    GAMMA = 0.9  # Discount factor
    MAX_POLICY_EVAL_ERROR = 10e-7  # Convergence threshold for policy evaluation
    MAX_POLICY_EVAL_ITER = 1000  # Maximum iterations for policy evaluation
    MAX_POLICY_ITER_ERROR = -0.5  # Convergence threshold for policy iteration
    MAX_POLICY_ITER_LOOP = 1000  # Maximum iterations for policy iteration

    # Create an instance of the GridWorldPolicyIteration class
    grid_world = GridWorldMDP(
        grid_width=GRID_WIDTH, 
        grid_height=GRID_HEIGHT,
        goal_states=GOAL_STATES,
        forbidden_states=FORBIDDEN_STATES,
        actions=ACTIONS,
        rewards=REWARDS
    )
    # Plot the grid world
    grid_world.plot_grid_world()

    # Run policy iteration
    value_states, policy = grid_world.policy_iteration(
        value_states=VALUE_STATES,
        policy=POLICY,
        gamma=GAMMA,
        max_policy_eval_error=MAX_POLICY_EVAL_ERROR,
        max_policy_eval_loop=MAX_POLICY_EVAL_ITER,
        max_policy_iter_error=MAX_POLICY_ITER_ERROR,
        max_policy_iter_loop=MAX_POLICY_ITER_LOOP
    )

    print("Final Value States:")
    for state, value in value_states.items():
        print(f"State {state}: {value:.2f}")
    
    print("\nFinal Policy:")
    for state, action in policy.items():
        print(f"State {state}: {action}")
