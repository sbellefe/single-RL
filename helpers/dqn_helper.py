import random
import torch as th

from collections import deque, namedtuple

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class ReplayMemory(object):
    transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def __init__(self, mem_capacity):
        self.memory = deque([], maxlen=mem_capacity)
        self.transition = namedtuple(
            'Transition',
            ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def pre_process(obs):
    state = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
    return state

def soft_update(Q, Q_prime, tau):
    """
        Softly updates the target agent network parameters.
        # θ′ ← τ θ + (1 −τ )θ′

        Args:
            Q (nn.Module): The policy network providing the new weights.
            Q_prime (nn.Module): The target network to update.
            tau (float): The soft update rate (0 < tau <= 1).
        """
    for Q_prime, policy_param in zip(Q_prime.parameters(), Q.parameters()):
        Q_prime.data.copy_(tau * policy_param.data + (1.0 - tau) * Q_prime.data
        )