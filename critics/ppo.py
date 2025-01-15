import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(PPOCritic, self).__init__()
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        x = state
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

    def critic_loss(self, values, old_values, returns, eps_clip):
        # value_clip = old_values + th.clamp(values - old_values, -eps_clip, eps_clip)
        value_clip = th.clamp(values, old_values - eps_clip, old_values + eps_clip)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        loss = th.max(loss_unclipped, loss_clipped).mean()
        return loss
