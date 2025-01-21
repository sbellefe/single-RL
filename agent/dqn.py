import random, sys
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F


class DQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # self.initialize_weights()

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')


    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     Q = self.fc3(x)
    #     return Q

    def select_action(self, state, epsilon, env):
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with th.no_grad():
                Q = self(state)
                greedy_action = Q.argmax(axis=1).cpu()
                # action = Q.max(1).indices.view(1, 1)
                # print(f"Action (GREEDY): {action}")
                return greedy_action
        else:
            # action = th.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=th.long)
            random_action = th.tensor([env.action_space.sample()], device=self.device)
            # action = th.tensor([env.action_space.sample()], device=self.device)
            # action = th.tensor([action], dtype=th.float, device=self.device)
            # print(f"Action (RANDOM): {action}")
            return random_action


        # global steps_done
        # sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #                 math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1
        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         # t.max(1) will return the largest column value of each row.
        #         # second column on max result is index of where max element was
        #         # found, so we pick action with the larger expected reward.
        #         return policy_net(state).max(1).indices.view(1, 1)
        # else:
        #     return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)