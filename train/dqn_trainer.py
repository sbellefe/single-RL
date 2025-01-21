import sys

import numpy as np
from copy import deepcopy
from itertools import count
import torch as th


from torch.distributions import Categorical

from agent.dqn_agent import QNetwork
from helpers.dqn_helper import ReplayMemory, pre_process, soft_update

class DQNtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device
        #initialize policy and target networks
        Q = QNetwork(params.state_dim, params.hidden_dim, params.action_dim).to(device)
        Q_prime = deepcopy(Q).to(device)

        opt = th.optim.Adam(Q.parameters(), lr=params.actor_lr, amsgrad=True)
        buffer = ReplayMemory(params.buffer_capacity)

        params.t_tot = 0
        episode_rewards = []
        test_rewards = []
        n_ep = 0

        for ep in range(params.total_train_episodes):
            obs, _ = env.reset()
            state = pre_process(obs)

            total_reward = 0

            for t in count():
                epsilon = params.epsilon
                action = Q.select_action(state, epsilon, env)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward = th.tensor([reward], device=device)
                next_state = None if terminated else pre_process(obs)
                # print(f"Action: {action.shape}\n")
                      # f"Action_tensor {pre_process(action).shape}")

                buffer.push(state, action, next_state, reward, done)

                total_reward += reward
                params.t_tot += 1
                state = next_state

                #run optimization if buffer is ready
                if len(buffer) >= params.train_start_delay:
                    # for _ in range(4):
                    self.optimize(buffer, Q, Q_prime, opt, params)
                if done:
                    break

            n_ep += 1
            episode_rewards.append(total_reward)

            if n_ep % params.test_interval == 0:
                test_reward = self.test(deepcopy(Q), deepcopy(env), params)
                test_rewards.append(test_reward)
                print(f'Test reward at episode {n_ep}: {test_reward:.2f} | '
                      f'Total steps: {params.t_tot} | '
                      f'Current training epsilon: {params.epsilon:.2f}')

            soft_update(Q, Q_prime, params.tau)

        print("Algorithm done")
        return episode_rewards, test_rewards


    def optimize(self, buffer, agent, target_agent, opt, params):
        device = params.device
        # Sample and unpack a batch of transitions from the replay buffer
        transitions = buffer.sample(params.batch_size)
        batch = ReplayMemory.transition(*zip(*transitions))


        #create mask for non-terminal states
        # non_final_mask = th.tensor((map(lambda s: s is not None, batch.next_state)), device=device, dtype=th.bool)
        non_final_mask = ~th.tensor(batch.done, device=device, dtype=th.bool)

        #collect non-terminal states
        # non_final_next_states = th.cat([s for s in batch.next_state if s is not None])
        non_final_next_states = th.cat([s_ for s_, done in zip(batch.next_state, batch.done) if not done]
)
        state_batch = th.cat(batch.state)
        # action_batch = th.cat([a for act in zip(batch.action) if not done])

        action_batch = th.cat(batch.action).unsqueeze(-1)
        # print(f"Actions: {batch.action}\n"
        #       f"Action: {batch.action[0].shape}\n"
        #       f"States: {batch.state[0].shape}\n"
        #       f"States_batch: {state_batch.shape}\n"
        #       f"Actions_batch: {action_batch.shape}\n")

        reward_batch = th.cat(batch.reward)

        # Compute Q(s_t, a)
        current_q_values = agent(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) using the target network
        next_q_values = th.zeros(params.batch_size, device=device)
        with th.no_grad():
            next_q_values[non_final_mask] = target_agent(non_final_next_states).max(1).values

        # Compute expected Q values
        target_q_values = (next_q_values * params.gamma) + reward_batch

        # Compute loss and optimize
        loss_function = th.nn.SmoothL1Loss()
        loss = loss_function(current_q_values, target_q_values.unsqueeze(1))

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=params.grad_norm_clip)
        opt.step()


    @staticmethod
    def test(agent, env, params):
        # env.render(mode="human")

        test_rewards = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            total_reward = 0
            obs, _ = env.reset()

            for t in range(params.t_max):
                state = pre_process(obs)
                action = agent.select_action(state, params.eps_test, env)
                next_obs, reward, done, trunc, _ = env.step(action.item())

                total_reward += reward
                obs = next_obs
                if done or trunc:
                    break

            test_rewards[i] = total_reward

        env.close()
        average_reward = np.mean(test_rewards)
        return average_reward