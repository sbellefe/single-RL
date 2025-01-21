from copy import deepcopy
import numpy as np
import torch as th
from torch.distributions import Categorical

from agent.ppo_actor import PPOActor
from critics.ppo_critic import PPOCritic
from helpers.ppo_helper import BatchProcessing, compute_GAE, pre_process

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class PPOtrainer:
    def __init__(self):
        pass

    def train(self, env, params):
        actor = PPOActor(params.state_dim, params.actor_hidden_dim, params.action_dim)
        critic = PPOCritic(params.state_dim, params.critic_hidden_dim)
        actor_opt = th.optim.Adam(actor.parameters(), lr=params.actor_lr)
        critic_opt = th.optim.Adam(critic.parameters(), lr=params.critic_lr)

        episode_rewards = []
        test_rewards = []

        n_ep = 0

        for it in range(params.train_iterations):
            buffer = []

            for ep in range(params.buffer_episodes):
                state_history = []
                action_history = []
                logp_history = []
                reward_history = []
                value_history = []
                done_history = []

                obs, _ = env.reset()
                total_reward = 0

                for t in range(params.t_max):
                    with th.no_grad():
                        state = pre_process(obs)
                        logits = actor(state)
                        action, logp, _ = actor.sample_action(logits)
                        value = critic(state)

                    next_obs, reward, done, truncated, _ = env.step(action.item())
                    done = done or truncated

                    state_history.append(state)
                    action_history.append(action)
                    logp_history.append(logp)
                    reward_history.append(reward)
                    value_history.append(value)
                    done_history.append(done)

                    total_reward += reward
                    obs = next_obs

                    if done:
                        break

                returns, advantages = compute_GAE(reward_history, value_history, done_history, params.gamma,
                                                  params.gae_lambda)

                episode_rewards.append(total_reward)
                n_ep += 1

                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                if n_ep % params.test_interval == 0:
                    test_reward = self.test(deepcopy(actor), deepcopy(env), params.test_episodes, params.t_max)
                    test_rewards.append(test_reward)
                    print(f'Test reward at episode {n_ep}: {test_reward:.2f}')

            batch_process = BatchProcessing()

            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages \
                = batch_process.collate_batch(buffer, device)

            dataset = th.utils.data.TensorDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns,
                                                  batch_advantages)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

            for _ in range(params.opt_epochs):
                for batch in dataloader:
                    states_mb, actions_mb, logp_mb, values_mb, returns_mb, advantages_mb = batch

                    states_mb = states_mb.to(device)
                    actions_mb = actions_mb.to(device)
                    logp_mb = logp_mb.to(device)
                    values_mb = values_mb.to(device)
                    returns_mb = returns_mb.to(device)
                    advantages_mb = advantages_mb.to(device)

                    # Critic Update
                    critic_opt.zero_grad()
                    values_new = critic(states_mb)
                    critic_loss = critic.critic_loss(values_new, values_mb, returns_mb, params.eps_clip)
                    critic_loss.backward()
                    critic_opt.step()

                    # Actor Update
                    actor_opt.zero_grad()
                    dist = Categorical(logits=actor(states_mb))
                    logp_new = dist.log_prob(actions_mb)
                    entropy = dist.entropy().mean()
                    actor_loss = actor.actor_loss(logp_new, logp_mb, advantages_mb, params.eps_clip)
                    actor_loss = actor_loss - params.entropy_coef * entropy
                    actor_loss.backward()
                    actor_opt.step()

        print("Algorithm done")
        return episode_rewards, test_rewards

    def test(self, actor, env, test_episodes, t_max):
        # env.render(mode="human")

        test_rewards = np.zeros(test_episodes)

        for i in range(test_episodes):
            total_reward = 0
            obs, _ = env.reset()

            for t in range(t_max):
                state = pre_process(obs)
                logits = actor(state)
                action, _, _ = actor.sample_action(logits)
                next_obs, reward, done, _, _ = env.step(action.item())

                total_reward += reward
                obs = next_obs
                if done:
                    break

            test_rewards[i] = total_reward

        env.close()
        average_reward = np.mean(test_rewards)
        return average_reward