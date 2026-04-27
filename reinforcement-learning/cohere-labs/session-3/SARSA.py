import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import random
import time
from typing import Any, Tuple, Dict

# ==========================================
# Phase 1: Training SARSA(lambda)
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
lam: float = 0.9            # Trace decay parameter (lambda)
epsilon: float = 0.1        # Exploration rate
num_episodes: int = 2000

def choose_action(state: int, Q: NDArray[np.float64], epsilon: float, env: gym.Env) -> int:
    """Returns an action based on an epsilon-greedy policy."""
    if random.uniform(0.0, 1.0) < epsilon:
        return int(env.action_space.sample())
    else:
        return int(np.argmax(Q[state, :]))

print("Training SARSA(lambda) started...")

for episode in range(num_episodes):
    # E(s,a) = 0, for all s, a (Reset eligibility traces at the start of each episode)
    E: NDArray[np.float64] = np.zeros([n_states, n_actions], dtype=np.float64)
    
    # Initialize S
    state_info: Tuple[int, Dict[str, Any]] = env.reset()
    state: int = state_info[0]
    
    # Choose A from S
    action: int = choose_action(state, Q, epsilon, env)
    
    done: bool = False
    
    while not done:
        # Take action A, observe R, S'
        next_state: int
        reward: float
        terminated: bool
        truncated: bool
        info: Dict[str, Any]
        
        next_state, reward, terminated, truncated, info = env.step(action) # type: ignore
        done = terminated or truncated
        
        # Choose A' from S'
        next_action: int = choose_action(next_state, Q, epsilon, env)
        
        # delta <- R + gamma * Q(S', A') - Q(S, A)
        td_target: float = float(reward) + gamma * Q[next_state, next_action]
        delta: float = td_target - Q[state, action]
        
        # E(S, A) <- E(S, A) + 1 (Accumulating trace)
        E[state, action] += 1.0
        
        # For all s, a: 
        # Q(s,a) <- Q(s,a) + alpha * delta * E(s,a)
        # E(s,a) <- gamma * lambda * E(s,a)
        # Note: We use NumPy vectorization here instead of nested loops for massive performance gains.
        Q += alpha * delta * E
        E *= gamma * lam
        
        # S <- S'; A <- A'
        state = next_state
        action = next_action

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