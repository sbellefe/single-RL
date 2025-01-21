from copy import deepcopy as copy
import numpy as np
import torch as th
from torch.distributions import Categorical
from agent.ppo_actor import PPOActor
from critics.ppo_critic import PPOCritic
from agent.a2c_actor import A2CActor
from critics.a2c import A2CCritic
from helpers.a2c_helper import to_tensor, pre_process
from helpers.a2c_bp import BatchTraining
from helpers.ppo_helper import BatchProcessing, compute_GAE, testPPO, pre_process

from util.logger import Figure
from util.a2c_logger import a2c_test
import sys

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


                state, _ = env.reset()
                total_reward = 0

                for t in range(params.t_max):
                    with th.no_grad():
                        state = pre_process(state)
                        logits = actor(state)
                        action, logp, _ = actor.action_sampler(logits)
                        value = critic(state)

                    next_state, reward, done, truncated, info = env.step(action.item())
                    done = done or truncated

                    state_history.append(state)
                    action_history.append(action)
                    logp_history.append(logp)
                    reward_history.append(reward)
                    value_history.append(value)
                    done_history.append(done)

                    total_reward += reward
                    state = next_state

                    if done:
                        break

                returns, advantages = compute_GAE(reward_history, value_history, done_history, params.gamma, params.gae_lambda)

                episode_rewards.append(total_reward)
                n_ep += 1

                buffer.append((state_history, action_history, logp_history, value_history, returns, advantages))

                if n_ep % params.test_interval == 0:
                    test_reward = testPPO(copy(actor), copy(env), params.test_episodes, params.t_max)
                    test_rewards.append(test_reward)
                    print(f'Test reward at episode {n_ep}: {test_reward:.2f}')

            batch_process = BatchProcessing()

            batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages \
                = batch_process.collate_batch(buffer, device)

            dataset = th.utils.data.TensorDataset(batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages)
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

                    #Critic Update
                    critic_opt.zero_grad()
                    values_new = critic(states_mb)
                    critic_loss = critic.critic_loss(values_new, values_mb, returns_mb, params.eps_clip)
                    critic_loss.backward()
                    critic_opt.step()

                    #Actor Update
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
            state, _ = env.reset()

            for t in range(t_max):
                state = pre_process(state)
                logits = actor(state)
                action, _, _ = actor.action_sampler(logits)
                next_state, reward, done, _, _ = env.step(action.item())

                total_reward += reward
                state = next_state
                if done:
                    break

            test_rewards[i] = total_reward

        env.close()
        average_reward = np.mean(test_rewards)
        return average_reward



class DQNtrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("DQNtrainer")

class A2Ctrainer:
    def __init__(self, env_name):
        self.env_name = env_name

    def train(self, env, env_name, state_dim, action_dim, action_offset ,gamma, actor_hidden_dim, critic_hidden_dim,
            value_dim, alpha, beta, num_training_episodes ,num_batch_episodes, t_max, tau,
            test_interval, num_test_episodes):
        
        actor = A2CActor(state_dim, action_dim, actor_hidden_dim)
        critic = A2CCritic(state_dim, critic_hidden_dim, value_dim)

        optimizer_actor = th.optim.Adam(actor.parameters(), lr=alpha)
        optimizer_critic = th.optim.Adam(critic.parameters(), lr=beta)

        episode_rewards = []
        test_rewards = []

        episode = 0

        for te in range(num_training_episodes):
            batch_buffer = []
            batch_rtrns = []
            for e in range(num_batch_episodes):
                state, _ = env.reset()
                prev_state = None
                total_reward = 0
                done = False
                t = 0
                buffer = []

                while not done and (t < t_max):

                    if 'PongNoFrameskip-v4' in self.env_name:
                        state_tensor = env.pre_process(state, prev_state)
                    else:
                        state_tensor = to_tensor(state)

                    logits = actor(state_tensor)
                    action, dist = actor.action_sampler(logits)
                    converted_action = action.item() + action_offset
                    next_state, reward, done, _, info = env.step(converted_action)
                    if 'PongNoFrameskip-v4' in self.env_name:
                        next_state_tensor = env.pre_process(next_state, state)
                    else:
                        next_state_tensor = to_tensor(next_state)

                    buffer.append((state_tensor, action, reward, next_state_tensor))
                    prev_state = state
                    state = next_state
                    total_reward += reward
                    t+=1

                episode_rewards.append(total_reward)
                episode += 1

                rtrns = []
                if done:
                    R = 0
                else:
                    R = critic(to_tensor(state)).detach().item()

                for _, _, reward, _ in reversed(buffer):
                    R = reward + gamma * R
                    rtrns.append(R)
                rtrns.reverse()

                batch_buffer.extend(buffer)
                batch_rtrns.extend(rtrns)

                if episode % test_interval == 0:
                    actor_state = actor.state_dict()

                    test_reward = a2c_test(env_name, actor, num_test_episodes, t_max, action_offset)
                    test_rewards.append(test_reward)
                    print(f'Test reward at episode {episode}: {test_reward:.2f}')
                    actor.load_state_dict(actor_state)

            batch_training = BatchTraining()
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_rtrns = batch_training.collate_batch(batch_buffer, batch_rtrns)

            # Update Critic
            optimizer_critic.zero_grad()
            V = critic(batch_states)
            # Critic Loss
            critic_loss = (batch_rtrns - V).pow(2).mean()
            critic_loss.backward()
            optimizer_critic.step()

            # Update Actor
            optimizer_actor.zero_grad()
            logits = actor(batch_states)
            action_dist = Categorical(logits=logits)
            log_probs = action_dist.log_prob(batch_actions)
            entropies = action_dist.entropy()
            V = critic(batch_states).detach()
            advantage = (batch_rtrns - V).detach()

            # Actor Loss
            actor_loss = -(log_probs * advantage).mean() - tau * entropies.mean()
            actor_loss.backward()
            optimizer_actor.step()

        print("Algorithm done")
        return episode_rewards, test_rewards
