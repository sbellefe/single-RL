import argparse, time
import gymnasium as gym

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Import envs and runners here
# from runner.ppo_runner import PPOrunner
# from runner.a2c_runner import A2Crunner

# Import runner, trainers, and parameters classes here
from runner.runner import ALGOrunner
from train.ppo_trainer import PPOtrainer
from train.dqn_trainer import DQNtrainer
from train.train import A2Ctrainer
from util.parameters import ParametersPPO, ParametersDQN, ParametersA2C



def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "pong", or "cartpole".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "dqn", "ppo", or "a2c".')
    args = parser.parse_args()

    if args.env == 'cartpole':
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
    elif args.env == 'pong':
        env_name = 'PongNoFrameskip-v4'
        raise ValueError("Pong environment not implemented.")
    else:
        raise ValueError("Environment name incorrect or found")
    
    if args.algo == 'ppo':
        params = ParametersPPO()
        trainer = lambda: PPOtrainer()
    elif args.algo == 'a2c':
        params = ParametersA2C()
        trainer = lambda: A2Ctrainer()
    elif args.algo == 'dqn':
        params = ParametersDQN()
        trainer = lambda: DQNtrainer()
    else:
        raise ValueError("Algorithm name incorrect or not found")

    #add environment specific parameters
    params.env_name = env_name
    # TODO: add logic for discrete vs. continuous spaces
    params.state_dim = env.observation_space.shape[0]
    params.action_dim = env.action_space.n

    runner = ALGOrunner(env, trainer)
    runner.run_experiment(params)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution Time: {execution_time} seconds")


