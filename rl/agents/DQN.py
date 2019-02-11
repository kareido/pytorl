import random
import torch
from ._Base import Base_Agent
import rl.networks as network


class DQN_Agent(Base_Agent):
    def __init__(self, 
                 device, 
                 q_net, 
                 target_net, 
                 loss, 
                 optimizer, 
                 replay, 
                ):
        self.device = device
        self.q_net = q_net
        self.target_net = target_net
        self.loss = loss
        self.optimizer = optimizer
        self.replay = replay
    
    
    def reset(self):
        self.replay.clear()
        self.set_device()
        for param_group in self.q_net.parameters():
            torch.nn.init.uniform_(param_group)
        self.update_target()
        self.target_net.eval()
    
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    
    def set_device(self):
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
    
    def next_action(self, thres, rand_act, curr_input=None):
        sample_val = random.random()
        if sample_val > thres and curr_input is not None:
            with torch.no_grad():
                curr_q_val = self.q_net(curr_input)
                return curr_q_val.argmax(1).view(1, 1)
        else:
            # guarantee the return value is legal
            return rand_act.view(1, 1).to(self.device)
        
        
    def optimize(self):
        pass
    
    
        