import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network(nn.Module):
    """
    wrap a specific network into Q-network
    
    input_size: a 3-d tuple (or equivalent Iterable) indicating the size of input
    
    num_actions: corresponding to the size of output
    
    backbone: a pytorch network, if not specified, using original DQN instead
    
    replace_fc: the name of last linear (i.e. 'fc' or 'last_linear'), if not 
    specified, using backbone's last output layer instead.
    """
    def __init__(self, input_size=None, num_actions=None, backbone=None, replace_fc=None):
        super(Q_Network, self).__init__()
        if input_size is not None and len(input_size) != 3:
            raise ValueError('invalid input_size specified')
        self.input_size, self.num_actions = input_size, num_actions
        if not backbone:
            self.network = backbone
            if replace_fc is not None:
                if num_actions is None:
                    raise ValueError('num_actions must be specified if enabling replace_fc')
                last_fc = getattr(self.network, replace_fc)
                last_fc_in = last_fc.__dict__['in_features']
                last_fc_out = last_fc.__dict__['out_features']
                if last_fc_out != num_actions:
                    last_fc = nn.Linear(last_fc_in, num_actions)
            if input_size is not None:
                self.check_forward()
            else:
                print('warning: skip precheck due to input_size not specified', flush=True)
                
        else:
            if not (input_size and num_actions):
                raise ValueError(
                        'must specify input_size and num_actions if backbone not specified')
            self.network = _Original_DQN(input_size, num_actions)
            self.check_forward()
   

    def forward(self, x):
        return self.network(x)
    
    def check_forward(self):
        mock = torch.zeros(1, *self.input_size)
        try:
            self.network(mock)
        except:
            raise ValueError(
                    'network forward failure, presumably due to invalid input_size')

            
            
class _Original_DQN(nn.Module):
    """
    this is the Q-network used in the original DQN paper
    
    input_size: a 3-d tuple (or equivalent Iterable) indicating the size of input
    
    num_actions: corresponding to the size of output
    """
    def __init__(self, input_size, num_actions):
        super(_Original_DQN, self).__init__()
        
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_size[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
    
    
    @property
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_size)).shape[1]
    
    
    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        return x   
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
