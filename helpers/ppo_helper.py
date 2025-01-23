import torch as th
import sys

def pre_process(obs):
    state = th.FloatTensor(obs).unsqueeze(0)
    return state

class BatchProcessing:
    def __init__(self):
        pass

    def collate_batch(self, buffer, device):
        """process buffer into batch tensors once buffer is full"""
        batch_states, batch_actions, batch_logp = [],[],[]
        batch_values, batch_returns, batch_advantages = [],[],[]

        for data in buffer:
            state, action, logp, value, rtrn, adv = data

            state = th.stack(state).to(device)
            action = th.stack(action).to(device)
            logp = th.stack(logp).to(device)
            value = th.stack(value).to(device)

            batch_states.append(state)
            batch_actions.append(action)
            batch_logp.append(logp)
            batch_values.append(value)
            batch_returns.append(rtrn)
            batch_advantages.append(adv)

        #convert to tensors
        batch_states = th.cat(batch_states, dim=0)
        batch_actions = th.cat(batch_actions, dim=0)
        batch_logp = th.cat(batch_logp, dim=0)
        batch_values = th.cat(batch_values, dim=0).squeeze(-1)
        batch_returns = th.cat(batch_returns, dim=0)
        batch_advantages = th.cat(batch_advantages, dim=0).squeeze(-1)

        # normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / batch_advantages.std()

        return batch_states, batch_actions, batch_logp, batch_values, batch_returns, batch_advantages

def compute_GAE(rewards, values, dones, gamma, gae_lambda, device):
    advantages, returns = [], []
    R, gae = 0, 0

    values = values + [th.zeros_like(values[0])]

    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

        R = rewards[t] + gamma * R * next_non_terminal
        returns.insert(0, R)

    returns = [th.tensor(agent_returns) for agent_returns in returns]
    returns = th.stack(returns).to(device)
    advantages = th.stack(advantages).to(device)

    return returns, advantages