import os, sys, time
import numpy as np
import torch as th
from util.benchmarker import Utils

from torch.profiler import profile, record_function, ProfilerActivity

class ALGOrunner():
    def __init__(self, env, trainer):
        self.env = env
        self.trainer = trainer

    def run_experiment(self, params, load_save_result=False):
        if load_save_result is False or not os.path.isfile('all_train_returns.npy') or not os.path.isfile(
                'all_test_returns.npy'):
            all_train_returns = []
            all_test_returns = []

            for trial in range(params.num_trials):
                print(f"Trial: {trial + 1}")
                trainer = self.trainer()

                # print(th.cuda.is_available())
                # print(th.cuda.current_device())
                # print(th.cuda.get_device_name(0))
                # # device = th.device('cpu')
                # x = th.randn(100000, 100000, device=params.device)
                # start = time.time()
                # y = th.mm(x,x)
                # end = time.time()
                # print(f"Time taken: {end-start}")
                # sys.exit()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #              on_trace_ready=th.profiler.tensorboard_trace_handler('./log')) as prof:
                    # with record_function('model_training'):
                train_rewards, test_rewards = trainer.train(self.env, params)
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                all_train_returns.append(train_rewards)
                all_test_returns.append(test_rewards)
            np.save('all_train_returns.npy', all_train_returns)
            np.save('all_test_returns.npy', all_test_returns)
        else:
            all_train_returns = np.load('all_train_returns.npy')
            all_test_returns = np.load('all_test_returns.npy')
        # print(f"All train returns: {all_train_returns}"
        #       f"\nAll test returns: {all_test_returns}")
        # sys.exit()
        utils = Utils()
        average_returns, max_return, max_return_ci, individual_returns = utils.benchmark_plot(all_train_returns,
                                                                                              all_test_returns,
                                                                                              params.test_interval)
        print(f"Average Return: {np.round(average_returns, 2)}")
        print(f"Max Return: {max_return}")
        print(f"Max Return 95% CI: {max_return_ci}")
        print(f"Individual Max Returns: {individual_returns}")
        print("Completed experiment")