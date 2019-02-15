import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_MLP(nn.Module):
    """
    simple mlp for 1-D observation input from envs like
    CartPole-v1 etc.
    
    Args:
        input_size: a 2-D tuple or equivalent Iterable, indicating the size 
                        of input, i.e, the obervation space
        num_actions: corresponding to the size of output
        hidden_size: size of hidden layer outputs
    """
    def __init__(self, input_size=None, num_actions=None, hidden_size=256):
        assert input_size and num_actions
        assert len(input_size) == 2, 'input_size must be a 2-D Iterable'
        super(Q_MLP, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size 
        self.fc1 = nn.Linear(input_size[0] * input_size[1], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
    
    def features(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc3(x))
        return x
        
        
    