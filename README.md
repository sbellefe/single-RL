# single-RL
## requirements: 
```pip install -r requirements.txt```

## agent: 
ppo.py: Contains PPOActor and PPOCritic classes.

dqn.py: Contains DQNetwork agent class.

a2c.py: Contains A2CActor and A2Critic classes.

## runner:
runner.py: Contains ALGORunner class and run_experiment method (used for all algorithms).

## trainer:
ppo_trainer.py: Contains PPOtrainer class.

dqn_trainer.py: Contains DQNtrainer class.

a2c_trainer.py: Contains trainer class.

## env:
Implemented using Gymnasium:
- CartPole-v1

## helper:

Contains additional classes and functions for training algorithms (e.g. replay buffer).

## util:

benchmarker.py: Plot training and testing results.

parameters.py: Hyperparameter classes for each algorithm.

## main.py:

Use this command in terminal to run code:

```python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE```\
e.g ```python main.py --env cartpole --algo ppo```