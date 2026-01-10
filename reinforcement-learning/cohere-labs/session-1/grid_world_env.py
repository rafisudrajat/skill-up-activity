from typing import Tuple, Set
try:
    from .defined_types import State, Action, Reward, GridSize
except ImportError:
    from defined_types import State, Action, Reward, GridSize



class GridWorldEnvironment():
    """
    A class representing a grid world environment for reinforcement learning.

    The environment consists of a grid of states where an agent can move
    up, down, left, or right. Some states are forbidden, and one state is the goal.
    """
    def __init__(self, grid_size: GridSize, forbidden_states: Set[State], goal_state: State):
        """
        Initialize the GridWorldEnvironment.

        Args:
            grid_size (GridSize): The size of the grid (rows, cols).
            forbidden_states (Set[State]): A set of states that yield a negative reward.
            goal_state (State): The target state that yields a positive reward.

        The states is in range y=[0, rows-1], x=[0,cols-1]
        """
        self.grid_size = grid_size
        self.forbidden_states = forbidden_states
        self.goal_state = goal_state

    def step(self, state:State, action: Action) -> Tuple[State, Reward]:
        """
        Take a step in the environment given a current state and action.

        Args:
            state (State): The current state (x, y).
            action (Action): The action to take ("up", "down", "left", "right").

        Returns:
            Tuple[State, Reward]: A tuple containing the next state and the reward received.
        """
        x, y = state
        x_next, y_next = x, y

        if action == "up" and y > 0:
            y_next = y-1
        elif action == "down" and y < self.grid_size.rows - 1:
            y_next = y+1
        elif action == "left" and x > 0:
            x_next = x-1
        elif action == "right" and x < self.grid_size.cols - 1:
            x_next = x+1
            
        reward = self.get_reward(state, action, (x_next, y_next))
        
        return (x_next, y_next), reward
    
    def get_reward(self, state:State, action:Action, next_state:State)-> Reward:
        """
        Calculate the reward for a transition.

        Args:
            state (State): The current state.
            action (Action): The action taken.
            next_state (State): The resulting state after the action.

        Returns:
            Reward: The reward value. -1 for boundaries or forbidden states,
                    1 for the goal state, and 0 otherwise.
        """
        x,y = state
        if y == 0 and action == "up":
            return -1
        elif y == self.grid_size.rows - 1 and action == "down":
            return -1
        elif x == 0 and action == "left":
            return -1
        elif x == self.grid_size.cols - 1 and action == "right":
            return -1
        elif next_state == self.goal_state:
            return 1
        elif next_state in self.forbidden_states:
            return -10
        else:
            return 0