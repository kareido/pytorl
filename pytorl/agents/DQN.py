import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from ._base_agent import Agent


class DQN_Agent(Agent):
    def __init__(self, 
                 device, 
                 q_net, 
                 target_net, 
                 loss_func, 
                 optimizer_func,  
                 replay, 
                ):
        self.device = device
        self.q_net = q_net
        self.target_net = target_net
        self.loss = loss_func
        self._get_optimizer = optimizer_func
        self.replay = replay
        self.batch_size = replay.batch_size
        # attributes for optimization
        self.lr = .0001
        self.gamma = .99
        self.optimize_freq = 1
        self.update_target_freq = 1
        # attributes for action selection
        self.get_sample = None
        self.get_thres = lambda: 0
        
    
    def set_exploration(self,
                        get_sample=None, 
                        get_thres=lambda: 0,  
                        ):
        self.get_sample = get_sample
        self.get_thres = get_thres
    
    
    def set_optimize_scheme(self,
                            lr=.0001, gamma=.99, 
                            optimize_freq=1, 
                            update_target_freq=1, 
                           ):
        # set attributes
        self.lr = lr
        self.gamma = gamma
        self.optimize_freq = optimize_freq
        self.update_target_freq = update_target_freq
        self.optimizer = self._get_optimizer(
                            self.q_net.parameters(), 
                            lr=self.lr)
    
    
    def reset(self):
        self.replay.clear()
        self.set_device()
        self.global_timesteps('set', 0)
        self.optimizer_timer('set', 0)
        for name, params in self.q_net.named_parameters():
            if 'bias' in name:
                # to avoid 'Fan in and fan out can not be computed for tensor with fewer 
                # than 2 dimensions' problem
                nn.init.zeros_(params)
            else:
                nn.init.kaiming_normal_(params)
        self.update_target()
        self.q_net.train(True)
        self.target_net.train(False)
        self.set_optimize_scheme()
        self.set_exploration()
    
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    
    def set_device(self):
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
    
    def next_action(self, get_state):
        if not hasattr(get_state, '__call__'): 
            curr_state = lambda: get_state
        else: 
            curr_state = get_state
        sample_val = random.random()
        if sample_val >= self.get_thres():
            with torch.no_grad():
                curr_q_val = self.q_net(get_state().to(self.device))
            return curr_q_val.argmax(1).item()
        else:
            return self.get_sample()
        
        
    def optimize(self):
        self.optimize_timer('add')
        if self.optimize_timer() % self.optimize_freq != 0: return
        self.global_timesteps('add')
        sample_exp = self.replay.sample()
        batch = self.replay.form_obj(*zip(*sample_exp))
        curr_states = torch.cat(batch.curr_state).to(self.device)
        actions = torch.tensor(batch.action).to(self.device).view(-1, 1)
        rewards = torch.tensor(batch.reward).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=self.device, dtype=torch.uint8)
        non_final_next = torch.cat(
                            [s for s in batch.next_state if s is not None]).to(self.device)
        predicted_q_values = self.q_net(curr_states).gather(1, actions)
        targeted_q_values = torch.zeros(rewards.shape[0], device=self.device)
        # compute Q values via stationary target network, this 'try' is to avoid the situation 
        # when all next states are None
        try:
            targeted_q_values[non_final_mask] = self.target_net(
                                                    non_final_next).max(1)[0].detach()
        except TypeError: print('encountered a case where all next states are None', flush=True)
        # compute the expected Q values
        expected_q_values = (targeted_q_values * self.gamma) + rewards
        # compute loss
        q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1))
        # optimize the model
        self.optimizer.zero_grad()
        q_net_loss.backward()
        clip_grad_value_(self.q_net.parameters(), 1)
        self.optimizer.step()
        # update target network
        if self.global_timesteps() % self.update_target_freq != 0:
            self.update_target()
        # tensorboard recording
        if self._tensorboard is not None:
            reward_mean = rewards.mean().item()
            predicted_q_values_mean = predicted_q_values.mean().item()
            expected_q_values_mean = expected_q_values.mean().item()

            self._tensorboard.add_scalar('timestep/replay_reward-mean', 
                                   reward_mean, self.global_timesteps())
            self._tensorboard.add_scalar('timestep/loss', q_net_loss, self.global_timesteps())
            self._tensorboard.add_scalar('timestep/predicted_q_values-mean', 
                                   predicted_q_values_mean, self.global_timesteps())
            self._tensorboard.add_scalar('timestep/expected_q_values-mean', 
                                   expected_q_values_mean, self.global_timesteps())

        
        
        
        
    
        