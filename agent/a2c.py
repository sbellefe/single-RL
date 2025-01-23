import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

class A2CActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
      super(A2CActor, self).__init__()

      # Define Layers
      self.fc1 = nn.Linear(state_dim, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, hidden_dim)
      self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Build network that maps state -> logits
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    @staticmethod
    def select_action(logits):
        #create distribution and sample action from it
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        return action, action_dist

class A2CCritic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(A2CCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = state # Build network that maps state -> value
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value