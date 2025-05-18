# gridworld-rl

This project implements reinforcement learning agents to navigate random, procedurally generated 2D gridworld environments, and to catch a player who is attempting to find and reach the escape.

## Project Structure

```
.
├── game.py                 # Script to run and visualize the game with a trained agent
├── gridworld_generator.py  # Script to generate Gridworld environments
├── README.md               # This file
├── train.py                # Script to train reinforcement learning agents
├── agents/                 # Contains different agent implementations
│   ├── dqn_agent.py        # Deep Q-Network (DQN) agent
│   └── qlearning_agent.py  # Q-learning agent
├── models/                 # Stores trained model checkpoints (e.g., for DQN agents)
├── plots/                  # Directory for storing performance plots (e.g., rewards over episodes)
└── q_tables/               # Directory for storing Q-tables for Q-learning agents
```
