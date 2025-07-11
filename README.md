**Note:** This repository is a personal fork of [Deepinlab2023/single-RL](https://github.com/Deepinlab2023/single-RL), originally developed under **Dr. Lei Lei** during my graduate research at the University of Guelph.
I contributed a significant rework of the repository with a focus on:
 - **Code clarity** and **maintainability**: streamlined the file structure and reduced redundant logic.
 - **Algorithm coverage**: added a missing DQN implementation and modified A2C.
 - **PPO improvements**: integrated Generalized Advantage Estimation (GAE), improved network weight initialization, and value loss clipping for better training stability.
 - **Framework modernization**: migrated from the deprecated `gym` package to `gymnasium`.
 - **Usability upgrades**: added a `requirements.txt` and updated the README for better documentation and reproducibility.
 - **Benchmarking**: completed test runs and added results.



# single-RL

Implementation of single-agent RL algorithms in CartPole environment.

## requirements: 
```pip install -r requirements.txt```

## agent: 
ppo.py: Contains ```PPOActor``` and ```PPOCritic``` classes.

a2c.py: Contains ```A2CActor``` and ```A2Critic``` classes.

dqn.py: Contains ```DQNetwork``` agent class.

## runner:
runner.py: Contains ```ALGORunner``` class and the method ```run_experiment``` used for all algorithms.

## trainer:
ppo_trainer.py: Contains ```PPOtrainer``` class with methods ```train```, ```test```.

a2c_trainer.py: Contains ```A2Ctrainer``` class with methods ```train```, ```test```.

dqn_trainer.py: Contains ```DQNtrainer``` class with methods ```train```, ```optimize```, ```test```. 



## helper:

Contains additional classes and functions for training algorithms (e.g. replay buffer, pre-processing...).

## env:
Implemented using Gymnasium:
- CartPole-v1

## util:

benchmarker.py: Compute evaluation metrics, plot training and testing results.

parameters.py: Hyperparameter classes for each algorithm.

## main.py:

Use this command in terminal to run code:

```python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE```\
e.g ```python main.py --env cartpole --algo ppo```
