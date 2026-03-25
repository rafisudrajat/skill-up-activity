import gymnasium as gym
from collections import defaultdict
from typing import Tuple, List, Set, DefaultDict, Optional

# --- Type Definitions ---
# Blackjack state: (Player current sum, Dealer's showing card, Usable Ace)
State = Tuple[int, int, int] 
# Action = 0: Stick, 1: Hit
Action = int
Reward = float
# A single step in an episode consists of the state, the action taken, and the resulting reward
EpisodeStep = Tuple[State, Action, Reward]
ValueFunction = DefaultDict[State, float]

def simple_policy(observation: State) -> Action:
    """Stick (0) if score is 20 or 21, otherwise Hit (1)."""
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def mc_prediction_first_visit(
    env: gym.Env, 
    num_episodes: int, 
    gamma: float = 1.0
) -> ValueFunction:

    returns_sum: DefaultDict[State, float] = defaultdict(float)
    returns_count: DefaultDict[State, int] = defaultdict(int)

    # 1. Gather all experience
    for i in range(num_episodes):
        episode: List[EpisodeStep] = []
        state: State
        state, _ = env.reset() 
        done: bool = False
        
        # Generate episode
        while not done:
            action: Action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, float(reward)))
            state = next_state
            
        # Evaluate episode
        states_in_episode: Set[State] = {step[0] for step in episode}
        
        for state_in_ep in states_in_episode:
            first_idx: int = next(i for i, step in enumerate(episode) if step[0] == state_in_ep)
            G: float = sum([step[2] * (gamma ** i) for i, step in enumerate(episode[first_idx:])])
            
            # Just accumulate the counts and sums
            returns_count[state_in_ep] += 1
            returns_sum[state_in_ep] += G

    # 2. Calculate the Value Function ONCE at the end
    V: ValueFunction = defaultdict(float)
    for state in returns_sum:
        V[state] = returns_sum[state] / returns_count[state]

    return V

def mc_prediction_every_visit(
    env: gym.Env, 
    num_episodes: int, 
    gamma: float = 1.0
) -> ValueFunction:
    
    returns_sum: DefaultDict[State, float] = defaultdict(float)
    returns_count: DefaultDict[State, int] = defaultdict(int)

    # 1. Gather all experience
    for i in range(num_episodes):
        episode: List[EpisodeStep] = []
        state: State
        state, _ = env.reset() 
        done: bool = False
        
        # Generate episode
        while not done:
            action: Action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, float(reward)))
            state = next_state
            
        # Evaluate episode (Every-Visit)
        for t, step_data in enumerate(episode):
            state_in_ep: State = step_data[0]
            
            # Calculate the total return (G) from THIS specific time-step 't' onward
            G: float = sum([step[2] * (gamma ** i) for i, step in enumerate(episode[t:])])
            
            # Increment counter and sum for EVERY visit 
            returns_count[state_in_ep] += 1
            returns_sum[state_in_ep] += G 

    # 2. Calculate the Value Function ONCE at the end
    V: ValueFunction = defaultdict(float)
    for state in returns_sum:
        V[state] = returns_sum[state] / returns_count[state]

    return V

def mc_prediction_incremental(
    env: gym.Env, 
    num_episodes: int, 
    gamma: float = 1.0,
    alpha: Optional[float] = None  # If None, it uses 1/N(s). If float, uses constant alpha.
) -> ValueFunction:
    """
    Every-visit MC using Incremental Updates to estimate V(s).
    """
    # We no longer need to track the sum of all returns!
    returns_count: DefaultDict[State, int] = defaultdict(int)
    
    # The value function V(s)
    V: ValueFunction = defaultdict(float)

    for i in range(num_episodes):
        # 1. Generate an episode using our policy
        episode: List[EpisodeStep] = []
        state: State
        state, _ = env.reset() 
        done: bool = False
        
        while not done:
            action: Action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, float(reward)))
            state = next_state
            
        # 2. Evaluate the episode and update incrementally
        # Looping through for each state S_t with return G_t
        for t, step_data in enumerate(episode):
            state_in_ep: State = step_data[0]
            
            # Calculate the total return (G_t) from time-step 't' onward
            G: float = sum([step[2] * (gamma ** i) for i, step in enumerate(episode[t:])])
            
            # Increment counter N(s) <- N(s) + 1
            returns_count[state_in_ep] += 1
            
            # --- THE INCREMENTAL UPDATE ---
            if alpha is None:
                # Standard empirical mean update: V(s) <- V(s) + (1/N(s)) * (G_t - V(s)) 
                step_size = 1.0 / returns_count[state_in_ep]
                V[state_in_ep] = V[state_in_ep] + step_size * (G - V[state_in_ep])
            else:
                # Constant alpha for non-stationary problems: V(s) <- V(s) + a(G_t - V(s)) 
                V[state_in_ep] = V[state_in_ep] + alpha * (G - V[state_in_ep])

    return V

# --- Execution ---
if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    test_state: State = (20, 10, 0)

    print("Running Optimized First-Visit MC Prediction...")
    value_function_fv_mc: ValueFunction = mc_prediction_first_visit(env, num_episodes=10000)
    estimated_value_fv_mc: float = value_function_fv_mc[test_state]
    print(f"Estimated value for holding {test_state[0]} against a dealer's open card {test_state[1]} (First-Visit MC): {estimated_value_fv_mc:.3f}")
    
    print("\nRunning Optimized Every-Visit MC Prediction...")
    value_function_ev_mc: ValueFunction = mc_prediction_every_visit(env, num_episodes=10000)
    estimated_value_ev_mc: float = value_function_ev_mc[test_state]
    print(f"Estimated value for holding {test_state[0]} against a dealer's open card {test_state[1]} (Every-Visit MC): {estimated_value_ev_mc:.3f}")

    print("\nRunning Incremental MC Prediction...")
    value_function_inc_mc: ValueFunction = mc_prediction_incremental(env, num_episodes=10000)
    estimated_value_inc_mc: float = value_function_inc_mc[test_state]
    print(f"Estimated value for holding {test_state[0]} against a dealer's open card {test_state[1]} (Incremental MC): {estimated_value_inc_mc:.3f}") 

    env.close()