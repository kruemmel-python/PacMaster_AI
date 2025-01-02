# Deep Reinforcement Learning for Pac-Man using DDQN with Prioritized Experience Replay

This project implements a Deep Reinforcement Learning (DRL) agent that learns to play Pac-Man using a Double Deep Q-Network (DDQN) algorithm with a prioritized experience replay buffer. The agent is trained using TensorFlow and Keras, and the game environment is built with Pygame.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Game Environment](#game-environment)
- [Reinforcement Learning Agent](#reinforcement-learning-agent)
- [Prioritized Replay Buffer](#prioritized-replay-buffer)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
This project aims to create an AI agent capable of playing Pac-Man by learning optimal actions through trial and error. The agent uses the Double Deep Q-Network (DDQN) algorithm, which is an enhancement over the traditional Deep Q-Network (DQN).  The DDQN helps in reducing overestimation of Q-values, thus leading to more stable learning. Additionally, the implementation utilizes a prioritized experience replay buffer, enabling the agent to learn more effectively from experiences with high errors.

## Features
- Implements a Pac-Man game environment using Pygame.
- Utilizes a DDQN agent implemented with TensorFlow and Keras.
- Implements a prioritized experience replay buffer for more efficient learning.
- Includes hyperparameters tuning for optimized performance.
- Visualizes the game during training for observation and debugging.
- Allows for saving and loading trained models.

## Dependencies
- Python 3.6+
- Pygame
- NumPy
- TensorFlow 2.x
- Keras

Install the required packages:

```bash
pip install pygame numpy tensorflow
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/kruemmel-python/PacMaster_AI.git
cd PacMaster_AI
```
2. Install dependencies (see "Dependencies" section above).

## Usage
To train the agent and play the Pac-Man game, run the main script:

```bash
python main.py
```

The training process will print episode information, and the game window will visualize the current gameplay during training.

## Game Environment
The game environment is implemented using Pygame. Key features include:
- Dynamic Pac-Man, ghost, pellet, and power-pellet positioning
- Ghost movement based on Pac-Man's location.
- Collision detection between Pac-Man and game objects.
- Power-up mode triggered by consuming power pellets.
- Reward system based on game actions.

The state of the game is represented as a 17-dimensional vector comprising:
- Normalized Pac-Man position
- Normalized ghost positions
- Relative positions of the nearest pellet and power pellet
- Normalized distances to each ghost (max 4)
- An indicator of whether Pac-Man is in power-up mode.

## Reinforcement Learning Agent
The agent is built using TensorFlow and Keras. It features:
- A Double Deep Q-Network (DDQN) architecture to approximate the Q-function.
- The network architecture includes shared dense layers with Layer Normalization, and separate value and advantage streams.
- Adam optimizer is used with a specified learning rate.
- Epsilon-greedy policy for exploration/exploitation trade-off.
- Target network to stabilize training.

## Prioritized Replay Buffer
The experience replay buffer uses a prioritized method that helps the agent learn more effectively:
- Experiences with higher Temporal-Difference (TD) errors are sampled more frequently.
- The priorities are updated after the agent learns from a batch of experiences.
- The sampling probability is defined by the priorities of each experience.
- The buffer includes a mechanism for annealing the importance sampling weights (beta) over time.

## Training
The agent is trained over a defined number of episodes, during which:
- The agent interacts with the environment, taking actions, and receiving rewards.
- The experiences are stored in a prioritized replay buffer.
- The agent learns from sampled experiences, updating the weights of its network.
- Target network weights are updated at specified intervals.
- The agent's model is saved after every episode.

## Hyperparameter Tuning
The script includes hyperparameter tuning for the prioritized replay buffer. It tests different combinations of `alpha` and `beta_start` parameters:
- `alpha` controls the prioritization of the replay buffer.
- `beta_start` controls the initial importance sampling weight.

The best-performing hyperparameters are printed after the tuning process.

## Results
The training process will output the average reward per episode and the duration it took.

## Future Improvements
- Implement a more robust state representation of the game environment.
- Test different neural network architectures and layer combinations.
- Add a visualization of the prioritized replay buffer data during training.
- Improve the speed of the Pygame environment to train faster.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

