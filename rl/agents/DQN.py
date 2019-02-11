import random
import torch
from torch.nn.utils import clip_grad_value_
from ._Base import Base_Agent
import rl.networks as network


class DQN_Agent(Base_Agent):
    def __init__(self, 
                 device, 
                 q_net, 
                 target_net, 
                 loss_func, 
                 optimizer_func, 
                 lr, 
                 replay, 
                ):
        self.device = device
        self.q_net = q_net
        self.target_net = target_net
        self.get_loss = loss_func
        self.optimizer_func = optimizer_func
        self.lr = lr
        self.replay = replay
        self.optimizer = self.optimizer_func(
                            self.q_net.parameters(), 
                            lr=self.lr)
        self.optimize_func = None
    
    def reset(self):
        self.replay.clear()
        self.set_device()
        for param_group in self.q_net.parameters():
            torch.nn.init.uniform_(param_group)
        self.update_target()
        self.q_net.train(True)
        self.target_net.train(False)
        self.optimizer = self.optimizer_func(
                            self.q_net.parameters(), 
                            lr=self.lr)
        self.optimize_func = None
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    
    def set_device(self):
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
    
    def next_action(self, thres, rand_act, curr_input=None):
        sample_val = random.random()
        if sample_val > thres and curr_input is not None:
            with torch.no_grad():
                curr_q_val = self.q_net(curr_input.to(self.device))
                return curr_q_val.argmax(1).view(1, 1).to('cpu')
        else:
            # guarantee the return value is legal
            return rand_act.view(1, 1).to('cpu')
        
    
    def set_optimize_func(self, batch_size, gamma):
        def optimize_func():
            if len(self.replay) < batch_size:
                return
            sample_exp = self.replay.sample(batch_size)
            batch = self.replay.zipper(*zip(*sample_exp))
            curr_state_batch = torch.cat(batch.curr_state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.tensor(batch.reward).to(self.device)
            state_action_values = self.q_net(curr_state_batch).gather(1, action_batch)
            next_state_values = self.target_net(curr_state_batch).max(1)[0].detach()
            # compute the expected Q values
            expected_state_action_values = (next_state_values * gamma) + reward_batch
            # compute Huber loss
            loss = self.get_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_value_(self.q_net.parameters(), 1)
            self.optimizer.step()
            
        self.optimize_func = optimize_func
    
        
    def optimize(self):
        if not self.optimize_func:
            raise ValueError('should call set_optimize_func first')
        self.optimize_func()
        
        
        
        
    
        