= Policy Evaluation and Policy Iteration

This project give an example about the policy evaluation and policy iteration algorithm to find the optimal 
policy of a given MDP.

== Problem Definition

In this project we will try to find the optimal value function and optimal policy for a robot exploration task in a grid. 
The robot will try to explore its environment with these criteria:

1. The orange cells represent forbidden areas.
2. The blue cell represents the target area.
3. The reward settings are $r_"boundary" = 1, r_"forbidden" = −10$ 
4. $r_"target" = 1$
5. Discount rate $gamma = 0.9$

The optimal state value and the grid environment are defined as follows:

#figure(
  image("img/policy1-2-sol.png", width: 50%, scaling: "smooth"),
  caption: [
    Policy and Grid Environment.
  ],
)

The demonstration of policy iteration and policy evaluation can be seen in the #link("./test")[test] folder.