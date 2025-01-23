import numpy as np
from copy import deepcopy
from itertools import count
import torch as th
import gymnasium as gym

from agent.dqn import DQNetwork
from helpers.dqn_helper import ReplayMemory, pre_process, soft_update

class DQNtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device

        #initialize policy and target networks, optimizer, buffer
        Q = DQNetwork(params.state_dim, params.hidden_dim, params.action_dim).to(device)
        Q_prime = deepcopy(Q)
        opt = th.optim.Adam(Q.parameters(), lr=params.actor_lr, amsgrad=True)
        buffer = ReplayMemory(params.buffer_capacity)

        params.t_tot = 0
        episode_rewards = []
        test_rewards = []
        n_ep = 0

        for ep in range(params.total_train_episodes):
            obs, _ = env.reset()
            state = pre_process(obs).to(device)

            total_reward = 0

            for t in count():
                action = Q.select_action(state, params, env)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward = th.tensor([reward])

                #add None marker for terminal next_states
                next_state = None if done else pre_process(obs).to(device)

                #store transition in buffer
                buffer.push(state, action, next_state, reward, done)

                total_reward += reward
                params.t_tot += 1
                state = next_state

                #run optimization after delay
                if len(buffer) >= params.train_start_delay:
                    self.optimize(buffer, Q, Q_prime, opt, params)

                #update target network weights
                soft_update(Q, Q_prime, params.tau)

                if done:
                    break

            n_ep += 1
            episode_rewards.append(total_reward)

            #test agent at interval and print results
            if n_ep % params.test_interval == 0:
                test_reward = self.test(deepcopy(Q), params)
                test_rewards.append(test_reward)
                print(f'Test reward at episode {n_ep}: {test_reward:.2f} (train reward: {total_reward.item():.2f}) | '
                      f'Total steps: {params.t_tot} | '
                      f'Current training epsilon: {params.epsilon:.2f}')

        print("Trial complete")
        return episode_rewards, test_rewards

    @staticmethod
    def optimize(buffer, Q, Q_prime, opt, params):
        device = params.device

        if len(buffer) < params.batch_size:
            return #skip optimization if buffer not long enough

        # Sample and unpack a batch of transitions from the replay buffer
        transitions = buffer.sample(params.batch_size)
        batch = ReplayMemory.transition(*zip(*transitions))

        # create mask for non-terminal states
        done_mask = ~th.tensor(batch.done, device=device, dtype=th.bool)

        #convert to tensors (next_states_batch excludes terminal states)
        state_batch = th.cat(batch.state)
        action_batch = th.cat(batch.action).to(device)
        reward_batch = th.cat(batch.reward).to(device)
        next_state_batch = th.cat([s_ for s_ in batch.next_state if s_ is not None])

        # Compute Q(s_t, a)
        current_q_values = Q(state_batch).gather(1, action_batch).to(device)

        # Compute V(s_{t+1}) using the target network
        next_q_values = th.zeros(params.batch_size, device=device)
        with th.no_grad():
            next_q_values[done_mask] = Q_prime(next_state_batch).max(1).values

        # Compute expected Q values
        target_q_values = (next_q_values * params.gamma) + reward_batch

        # Compute loss and optimize
        loss_function = th.nn.SmoothL1Loss()
        loss = loss_function(current_q_values, target_q_values.unsqueeze(1))

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_value_(Q.parameters(), params.grad_clip)
        opt.step()


    @staticmethod
    def test(agent, params):
        """tests agent and averages result, add keyword render_mode="human"
            to the line below to watch testing"""
        test_env = gym.make(params.env_name) #, render_mode="human")

        test_rewards = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            total_reward = 0
            obs, _ = test_env.reset()

            for t in range(params.t_max):
                state = pre_process(obs)
                action = agent.select_action(state, params, test_env, testing=True)
                next_obs, reward, done, trunc, _ = test_env.step(action.item())

                total_reward += reward
                obs = next_obs
                if done or trunc:
                    break

            test_rewards[i] = total_reward

        test_env.close()
        average_reward = np.mean(test_rewards)
        return average_reward