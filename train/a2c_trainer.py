import numpy as np
import torch as th
import gymnasium as gym
from copy import deepcopy
from torch.distributions import Categorical

from agent.a2c import A2CActor, A2CCritic
from helpers.a2c_helper import BatchTraining, pre_process


class A2Ctrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        device = params.device
        actor = A2CActor(params.state_dim, params.action_dim, params.actor_hidden_dim).to(device)
        critic = A2CCritic(params.state_dim, params.critic_hidden_dim).to(device)

        actor_opt = th.optim.Adam(actor.parameters(), lr=params.actor_lr)
        critic_opt = th.optim.Adam(critic.parameters(), lr=params.critic_lr)

        episode_rewards = []
        test_rewards = []

        n_ep = 0

        for it in range(params.train_iterations):
            obs, _ = env.reset()
            batch_buffer = []
            batch_rtrns = []
            for ep in range(params.batch_size):
                obs, _ = env.reset()
                total_reward = 0
                done = False
                t = 0
                buffer = []

                while not done and (t < params.t_max):
                    state = pre_process(obs).to(device)

                    logits = actor(state)
                    action, dist = actor.select_action(logits)
                    next_obs, reward, done, truncated, _ = env.step(action.item())
                    done = done or truncated
                    next_state = pre_process(next_obs).to(device)

                    buffer.append((state, action, reward, next_state))
                    obs = next_obs
                    total_reward += reward
                    t += 1

                n_ep += 1
                episode_rewards.append(total_reward)

                #Compute returns for episode
                rtrns = []
                if done:
                    R = 0
                else:
                    R = critic(state).detach().item()

                for _, _, reward, _ in reversed(buffer):
                    R = reward + params.gamma * R
                    rtrns.append(R)
                rtrns.reverse()

                #store episode and computed returns in buffer
                batch_buffer.extend(buffer)
                batch_rtrns.extend(rtrns)

                if n_ep % params.test_interval == 0:
                    test_reward = self.test(deepcopy(actor), params)
                    test_rewards.append(test_reward)
                    print(f'Test reward at episode {n_ep}: {test_reward:.2f} '
                          f'(train reward: {total_reward:.2f})')

            batch_training = BatchTraining()
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_rtrns = batch_training.collate_batch(
                batch_buffer, batch_rtrns)

            # Actor Update
            actor_opt.zero_grad()
            logits = actor(batch_states)
            action_dist = Categorical(logits=logits)
            log_probs = action_dist.log_prob(batch_actions)
            entropies = action_dist.entropy()
            V = critic(batch_states).detach()
            advantage = (batch_rtrns - V).detach()
            actor_loss = -(log_probs * advantage).mean() - params.entropy_coef * entropies.mean()
            actor_loss.backward()
            actor_opt.step()

            # Critic Update
            critic_opt.zero_grad()
            V = critic(batch_states)
            critic_loss = (batch_rtrns - V).pow(2).mean()
            critic_loss.backward()
            critic_opt.step()

        print("Trial done")
        return episode_rewards, test_rewards

    @staticmethod
    def test(actor, params):
        """tests agent and averages result, add keyword render_mode="human"
            to the line below to watch testing"""
        test_env = gym.make(params.env_name)  # , render_mode="human")

        test_rewards = np.zeros(params.test_episodes)

        for i in range(params.test_episodes):
            total_reward = 0
            obs, _ = test_env.reset()

            for t in range(params.t_max):
                state = pre_process(obs)
                logits = actor(state)
                action, _ = actor.select_action(logits)
                next_obs, reward, done, trunc, _ = test_env.step(action.item())

                total_reward += reward
                obs = next_obs
                if done or trunc:
                    break

            test_rewards[i] = total_reward

        test_env.close()
        average_reward = np.mean(test_rewards)
        return average_reward