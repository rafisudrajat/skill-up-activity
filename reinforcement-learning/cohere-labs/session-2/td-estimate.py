import gymnasium as gym
from collections import defaultdict
from typing import Tuple, DefaultDict, Optional

# --- Type Definitions ---
# Blackjack state: (Player current sum, Dealer's showing card, Usable Ace)
State = Tuple[int, int, int] 
# Action = 0: Stick, 1: Hit
Action = int
Reward = float
ValueFunction = DefaultDict[State, float]
TraceFunction = DefaultDict[State, float]

def simple_policy(observation: State) -> Action:
    """Stick (0) if score is 20 or 21, otherwise Hit (1)."""
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def td_zero_prediction(
    env: gym.Env, 
    num_episodes: int, 
    alpha: Optional[float] = None,  # If None, uses 1/N(s). If float, uses constant alpha.
    gamma: float = 1.0   # Discount factor
) -> ValueFunction:
    """
    Temporal Difference TD(0) Evaluation to estimate V(s).
    """
    # The value function V(s)
    V: ValueFunction = defaultdict(float)
    
    # We must track visits to calculate 1/N(s)
    visit_count: DefaultDict[State, int] = defaultdict(int)

    for i in range(num_episodes):
        state: State
        state, _ = env.reset() 
        done: bool = False
        
        while not done:
            # 1. Choose action according to policy
            action: Action = simple_policy(state)
            
            # 2. Take step in the environment
            next_state: State
            reward: Reward
            terminated: bool
            truncated: bool
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Increment the visit count for the current state
            visit_count[state] += 1
            
            # 4. Determine the step-size (alpha)
            current_alpha: float = alpha if alpha is not None else (1.0 / visit_count[state])
            
            # 5. Calculate TD Target
            if done:
                td_target: float = float(reward)
            else:
                td_target: float = float(reward) + gamma * V[next_state]
                
            # 6. Calculate TD Error
            td_error: float = td_target - V[state]
            
            # 7. Update Value Function: V(S_t) <- V(S_t) + alpha * TD_Error
            V[state] += current_alpha * td_error
            
            # 8. Move to the next state
            state = next_state

    return V

def td_lambda_prediction(
    env: gym.Env, 
    num_episodes: int, 
    lambda_: float = 0.5, # Trace decay parameter (lambda is a protected keyword in Python)
    alpha: float = 0.05,  # Learning rate / step-size
    gamma: float = 1.0    # Discount factor
) -> ValueFunction:
    """
    Backward View TD(λ) Evaluation to estimate V(s) using Eligibility Traces.
    """
    # The value function V(s)
    V: ValueFunction = defaultdict(float)

    for i in range(num_episodes):
        # E_0(s) = 0. We reset the eligibility trace at the start of EVERY episode.
        E: TraceFunction = defaultdict(float)
        
        state: State
        state, _ = env.reset() 
        done: bool = False
        
        while not done:
            # 1. Choose action according to policy
            action: Action = simple_policy(state)
            
            # 2. Take step in the environment
            next_state: State
            reward: Reward
            terminated: bool
            truncated: bool
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Calculate TD Target and TD Error (delta_t)
            if done:
                td_target: float = float(reward)
            else:
                td_target: float = float(reward) + gamma * V[next_state]
                
            # delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)
            td_error: float = td_target - V[state]
            
            # 4. Increment the Eligibility Trace for the CURRENT state
            # E_t(s) = gamma * lambda * E_{t-1}(s) + 1(S_t = s)
            # (We add 1 here, and do the decay for all states in the loop below)
            E[state] += 1.0
            
            # 5. "Shout Backward": Update ALL states we have visited so far in this episode
            # We iterate through keys() list to safely update the dictionary
            for visited_state in list(E.keys()):
                # Update Value: V(s) <- V(s) + alpha * delta_t * E_t(s)
                V[visited_state] += alpha * td_error * E[visited_state]
                
                # Decay Trace: Prepare E_{t-1} for the next time step
                E[visited_state] *= gamma * lambda_
                
                # Optional Optimization: If a trace gets extremely close to 0, 
                # we can delete it from the dictionary to save loop iterations.
                if E[visited_state] < 1e-4:
                    del E[visited_state]
            
            # 6. Move to the next state
            state = next_state

    return V

# --- Execution ---
if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    
    test_state: State = (20, 10, 0)

    print("Running TD(0) Prediction with 1/N(s) step-size...")
    v_td_zero_standard: ValueFunction = td_zero_prediction(env, num_episodes=50000)
    print(f"Estimated value (Hold 20 vs 10) using td zero with 1/N(s) alpha: {v_td_zero_standard[test_state]:.3f}")
    
    print("\nRunning TD(0) Prediction with constant Alpha = 0.03...")
    v_td_zero_constant: ValueFunction = td_zero_prediction(env, num_episodes=50000, alpha=0.03)
    print(f"Estimated value (Hold 20 vs 10) using td zero with constant alpha: {v_td_zero_constant[test_state]:.3f}")

    print("\nRunning TD(lambda) Prediction...")
    # lambda_ = 1.0 is equal to Monte Carlo, lambda_ = 0.0 is equal to TD(0) 
    # We use lambda_ = 0.5 to blend them.
    v_td_lambda: ValueFunction = td_lambda_prediction(
        env, num_episodes=50000, lambda_=0.5, alpha=0.05
    )
    print(f"Estimated value (Hold 20 vs 10) using td lambda: {v_td_lambda[test_state]:.3f}")

    
    env.close()