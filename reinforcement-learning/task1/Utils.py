from typing import Dict,List,Set
import numpy as np

# Define r_pi(s) for each state: is the expected reward for being in state s under policy pi.
def r_pi(prob_transitions:Dict[str,Set[str]],rewards:Dict[str,int],state:str)->int:
    next_states = prob_transitions[state]
    expected_reward = 0
    for ns in next_states:
        expected_reward += rewards[ns]
    return expected_reward

def update_state_values(prob_transitions:Dict[str,Set[str]],
                        rewards:Dict[str,int],
                        state_values:Dict[str,int],
                        states:List[str], 
                        gamma:float=0.9)->Dict[str,int]:
    """
    Update state values using the Bellman equation. 
    The Bellman equation is a recursive equation that relates the value of a state to the values of its successor states.
    
    """
    current_state_values_vector = np.array(list(state_values.values()))
    # Calculate the expected value of each state under the policy pi.
    r_pi_values = [r_pi(prob_transitions,rewards,state) for state in states]
    r_pi_values = np.array(r_pi_values)

    P_pi_values = [[1 if next_state in prob_transitions[prev_state] else 0 for next_state in states] for prev_state in states]
    P_pi_values = np.array(P_pi_values)
    updated_state_values_vector = P_pi_values@current_state_values_vector * gamma + r_pi_values
    new_state_values = {state: updated_state_values_vector[i] for i, state in enumerate(states)}

    return new_state_values

# Update state values until convergence.
def value_iteration(state_values:Dict[str,int],
                    prob_transitions:Dict[str,Set[str]],
                    rewards:Dict[str,int],
                    states:List[str],
                    gamma:float=0.9, 
                    tol:float=1e-5)->Dict[str,int]:
    """
    Perform value iteration to find the optimal state values.
    """
    while True:
        new_state_values = update_state_values(prob_transitions, rewards ,state_values, states, gamma)
        delta = np.max(np.abs(np.array(list(new_state_values.values())) - np.array(list(state_values.values()))))
        state_values = new_state_values
        if delta < tol:
            return state_values
