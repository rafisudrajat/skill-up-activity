from typing import NamedTuple, Tuple, Dict
from enum import Enum


class ActionSpace(str, Enum):
    up = "up"
    down = "down"
    right = "right"
    left = "left"
    none = ""


State = Tuple[int, int]  # Representing the agent's position in a grid
Action = ActionSpace
Reward = float
Policy = Dict[State,Action]
ValueFunction = Dict[State,float]

class TransitionFunction(NamedTuple):
    state:State
    nextState:State
    action:Action
    prob:float

class GridSize(NamedTuple):
    rows: int
    cols: int
