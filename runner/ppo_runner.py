import os
import numpy as np
from util.benchmarker import Utils
from util.parameters import ParametersPPO
from train.train import PPOtrainer
import gymnasium as gym

class PPOrunner():
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

    def run_experiment(self):
        params = ParametersPPO()
        if params.load_save_result is False or not os.path.isfile('all_train_returns.npy') or not os.path.isfile('all_test_returns.npy'):
            all_train_returns = []
            all_test_returns = []

            for trial in range(params.num_trials):
                print(f"Trial: {trial+1}")
                trainer = PPOtrainer(self.env)

                train_rewards, test_rewards = trainer.train(self.env, params, self.env_name)
                all_train_returns.append(train_rewards)
                all_test_returns.append(test_rewards)
            np.save('all_train_returns.npy', all_train_returns)
            np.save('all_test_returns.npy', all_test_returns)
        else:
            all_train_returns = np.load('all_train_returns.npy')
            all_test_returns = np.load('all_test_returns.npy')

        utils = Utils()
        average_returns, max_return, max_return_ci, individual_returns = utils.benchmark_plot(all_train_returns, all_test_returns, params.test_interval)
        print(f"Average Return: {np.round(average_returns,2)}")
        print(f"Max Return: {max_return}")
        print(f"Max Return 95% CI: {max_return_ci}")
        print(f"Individual Max Returns: {individual_returns}")
        print("Completed experiment")        

