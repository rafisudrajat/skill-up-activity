from Utils import *
from typing import Dict,List,Set


# Define all states
# In this example, we have 25 states: S1, S2, S3, ..., S25
# S1 represent the top-left corner and S25 represent the bottom-right corner of the grids.
# Forbidden states are S7, S8, S13, S17, S19, S22.
# Goal state is S18.
states:List[str] = [f"S{i}" for i in range(1, 26)]
forbidden_states:List[str] = ["S7", "S8", "S13", "S17", "S19", "S22"]
goal_state:str = "S18"

# Define all rewards
# In this example, we have a reward of 0 for each step taken, and a reward of -1 for reaching boundaries and forbidden states.
# The goal state S18 has a reward of +1.
rewards:Dict[str,int] = {state: 0 for state in states}
rewards[goal_state] = 1  # Goal state reward
for state in forbidden_states:
    rewards[state] = -1  # Forbidden state reward

# All transitions are deterministic, meaning that the next state is always the same given the current state and action.
# Define p_{pi}(s'|s) for each state-action pair: is the probability of transitioning from s to s' under policy pi.
prob_transitions:Dict[str,Set[str]] = {
    "S1": {"S2"},
    "S2": {"S3"},
    "S3": {"S4"},
    "S4": {"S9"},
    "S5": {"S10"},
    "S6": {"S1"},
    "S7": {"S2"},
    "S8": {"S9"},
    "S9": {"S14"},
    "S10": {"S15"},
    "S11": {"S6"},
    "S12": {"S11"},
    "S13": {"S18"},
    "S14": {"S15"},
    "S15": {"S20"},
    "S16": {"S11"},
    "S17": {"S18"},
    "S18": {"S18"},
    "S19": {"S18"},
    "S20": {"S25"},
    "S21": {"S16"},
    "S22": {"S23"},
    "S23": {"S18"},
    "S24": {"S23"},
    "S25": {"S24"},
}

# Initialize all state values to 0.
state_values:Dict[str,int] = {state: 0 for state in states}
print("Initial state values:", state_values)


state_values = value_iteration(state_values, prob_transitions, rewards, states ,tol=1e-7)
print()
print("Final state values:", state_values)