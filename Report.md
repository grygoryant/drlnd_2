[image1]: img/nets.png "Actor and Critic nets"
[image2]: img/ddpg.svg "DDPG"
[image3]: img/rewards.png "Average rewards"

# Report

## Learning algorithm

In order to solve this environment I've chosen Deep Deterministic Policy Gradients algorithm. At first, I've been trying to solve the first version of the environment. My base implementation of that method had two different neural network architecures: one for the Actor, and one for the Critic. It also uses a replay buffer and soft target networks updates.

![Actor and Critic nets][image1]

![DDPG][image2]

After solving the first version of the environment, I've decided to try the second one. In order order to do that, I had to modify the base DDPG implementation to use all agents for speeding up the experience gathering.

First of all, shared replay buffer has been implemented: each agent's environment observation is buffered into common replay buffer. After 10000 warmup steps each agent is able to sample environment states randomly (with the batch size of 128) from that buffer to learn. Each agent learns each 20 time steps for 10 times (i.e. gets 10 soft updates with a factor of 0.001 for their target nets). Adam optimizer has been used for both Actor and Critic nets with learning rates 0.0001 and 0.001 respectively. I've used the discount factor 0.99 for training.

Lastly, in the end of each episode the best actor is detected using it's score, and if it's better, then the last known best agent, all agents are substituted with the best agent according to the episode results.

It's also necessary no mention that Ornstein–Uhlenbeck process has been used to add noise to resulting actions during learning to make agent explore the environment, and it's important to reset that process's state after each episode.

## Rewards

Using the described algorithm I've been able to solve the environment just in 110 episodes, getting the following average rewards:

![Average rewards][image3]

Trained agents are getting the evaluation average of approximately 38.5 points. 

## Future work

Shortly after implementing my algorithm I've figured out that D4PG exists, and brief paper study gave me the understanding that it looks somehow similar to my flow, except shared buffer and best agent tracking. That's why I really want to try the original D4PG and to compare both implementations.