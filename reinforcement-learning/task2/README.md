# A project to solve Bellman Optimality Equation (BOE) to find the optimal policy and optimal state value

In this project we will use value iteration and policy iteration (truncated policy iteration) to solve BOE

Elementwise form of BOE

$$v(s) = \max_{\pi(s) \in \Pi(s)} \sum_{a \in A} \pi(a|s) \left( \sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v(s') \right), \quad s \in \mathcal{S}. \tag{1}$$

In matrix form, the above equation can be written as

$$v = \max_{\pi \in \Pi} (r_{\pi}+\gamma P_{\pi}v) \tag{2} $$

$$[r_{\pi}]_{s} \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a) r, \quad [P_{\pi}]_{s,s^{\prime}} = p(s^{\prime}|s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s) p(s^{\prime}|s,a)$$

Using value iteration and policy iteration, we will show that the initial policy for a robot exploration problem in 5x5 grids world with these criteria:
1. The orange cells represent forbidden areas.
2. The blue cell represents the target area.
3. The reward settings are $r_{boundary} = −1$ and $r_{forbidden} = -10$
4. $r_{target} = 1$
5. Discount rate $\gamma = 0.9$ 

can reach the optimal policy of

<img src="./img/optimal_policy.png" alt="drawing" width="500"/>