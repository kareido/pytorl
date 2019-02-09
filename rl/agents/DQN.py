import torch
import rl
import rl.networks as network
from ._Base import Base_Agent


class DQN_Agent(Base_Agent):
    def __init__(self,
                 q_net, 
                 target_net, 
                 optimizer, 
                 memory,
                ):
        self.q_net = q_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory
        
    
                 