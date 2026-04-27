import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import random
import time
from typing import Any, Tuple, Dict

# ==========================================
# Phase 1: Training Q-Learning
# ==========================================

# Initialize training environment
env: gym.Env = gym.make('Taxi-v3')

n_states: int = int(env.observation_space.n)  # type: ignore
print("Number of states:", n_states)
n_actions: int = int(env.action_space.n)      # type: ignore
print("Number of actions:", n_actions)

# Initialize Q(s, a) arbitrarily
Q: NDArray[np.float64] = np.zeros([n_states, n_actions], dtype=np.float64) 

# Hyperparameters
alpha: float = 0.1          # Learning rate
gamma: float = 0.99         # Discount factor
epsilon: float = 0.1        # Exploration rate
num_episodes: int = 2000

def choose_action(state: int, Q: NDArray[np.float64], epsilon: float, env: gym.Env) -> int:
    """Returns an action based on an epsilon-greedy policy."""
    if random.uniform(0.0, 1.0) < epsilon:
        return int(env.action_space.sample())
    else:
        return int(np.argmax(Q[state, :]))

print("Training Q-Learning started...")

for episode in range(num_episodes):
    # Initialize S
    state_info: Tuple[int, Dict[str, Any]] = env.reset()
    state: int = state_info[0]
    
    done: bool = False
    
    while not done:
        # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        action: int = choose_action(state, Q, epsilon, env)
        
        # Take action A, observe R, S'
        next_state: int
        reward: float
        terminated: bool
        truncated: bool
        info: Dict[str, Any]
        
        next_state, reward, terminated, truncated, info = env.step(action) # type: ignore
        done = terminated or truncated
        
        # Q-Learning Update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        # We find the maximum Q-value for the next state (Off-policy learning)
        best_next_action: int = int(np.argmax(Q[next_state, :]))
        td_target: float = float(reward) + gamma * Q[next_state, best_next_action]
        delta: float = td_target - Q[state, action]
        
        Q[state, action] += alpha * delta
        
        # S <- S'
        # Note: We do NOT update A <- A' here like we do in SARSA. 
        # In Q-learning, the next action is selected fresh at the top of the loop.
        state = next_state

env.close()
print("Training finished.")

# ==========================================
# Phase 2: Evaluation (With Rendering)
# ==========================================
print("Starting visual evaluation...")

eval_env: gym.Env = gym.make('Taxi-v3', render_mode='human')
eval_state_info: Tuple[int, Dict[str, Any]] = eval_env.reset()
eval_state: int = eval_state_info[0]
eval_done: bool = False
total_reward: float = 0.0

while not eval_done:
    # During evaluation, we use a fully greedy policy (epsilon = 0)
    eval_action: int = int(np.argmax(Q[eval_state, :]))
    
    eval_next_state: int
    eval_reward: float
    eval_terminated: bool
    eval_truncated: bool
    eval_info: Dict[str, Any]
    
    eval_next_state, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action) # type: ignore
    eval_done = eval_terminated or eval_truncated
    
    total_reward += float(eval_reward)
    eval_state = eval_next_state
    
    time.sleep(0.3)

print(f"Total reward from visual evaluation episode: {total_reward}")
eval_env.close()