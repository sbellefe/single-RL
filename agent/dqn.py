import random, sys
import torch as th
from torch import nn
from torch.nn import functional as F


class DQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q

    def select_action(self, state, params, env, testing=False):
        """" note that current training epsilon is calculated
              dynamically as an attribute of class ParametersDQN """
        epsilon = params.eps_test if testing else params.epsilon
        sample = random.random()

        # Epsilon-greedy action selection
        if sample > epsilon:
            with th.no_grad():
                Q_values = self(state)
                greedy_action = Q_values.max(1).indices.view(1,1)
                return greedy_action
        else:
            random_action = th.tensor([[env.action_space.sample()]], device=params.device, dtype=th.long)
            return random_action

