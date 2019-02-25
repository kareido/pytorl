import copy
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.distributed as dist
import pytorl.distributed as rl_dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .DQN import PrioritizedDQN_Agent


class GorilaDQN_Agent(PrioritizedDQN_Agent):
    def __init__(
        self, 
        device, 
        q_net, 
        target_net=None, 
        loss_func=None, 
        optimizer_func=None, 
        double_dqn=True, 
        momentum=0.1
        ):
        super(GorilaDQN_Agent, self).__init__(
             device, 
             q_net, 
             target_net=target_net, 
             loss_func=loss_func, 
             optimizer_func=optimizer_func, 
             double_dqn=double_dqn, 
        )
        self.replay = None
        if double_dqn:
            self._non_final_targeted_q_values = self._double_dqn_q_values
        else:
            self._non_final_targeted_q_values = self._natural_dqn_q_values
        
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.master_rank = rl_dist.get_master_rank()
        self._gradient_counter = 0
        self._gradient_timer = 0
        self.loss_running_mean = 0
        self.loss_running_std = 0
        self.momentum = momentum
        self.param_vector = parameters_to_vector(self.q_net.parameters()).zero_().detach()
        self.param_rank = torch.arange(len(self.param_vector)).to(self.device)
        self.padded_len = (len(self.param_rank) + self.world_size - 1) // (self.world_size - 1)
        if self.rank != self.master_rank:
            self.shard = self.rank if self.rank <= self.master_rank else self.rank - 1
            self.shard_idx = self.param_rank[(self.param_rank % (self.world_size - 1)) == self.shard]
            if len(self.shard_idx) < self.padded_len:
                self.shard_idx = torch.cat(
                    (self.shard_idx, torch.tensor([self.shard]).to(self.device)))
                assert len(self.shard_idx) == self.padded_len
            self.gradient = self.param_vector[self.shard_idx]
        else:
            self.shard_list = \
                list(range(self.master_rank)) + list(range(self.master_rank, self.world_size - 1))
            self.shard_idx = [None] * (self.world_size - 1)
            for idx in self.shard_list:
                self.shard_idx[idx] = self.param_rank[(self.param_rank % (self.world_size - 1)) == idx]
                if len(self.shard_idx[idx]) < self.padded_len:
                    self.shard_idx[idx] = torch.cat(
                        (self.shard_idx[idx], torch.tensor([idx]).to(self.device)))
                    assert len(self.shard_idx[idx]) == self.padded_len
            self.gradient = copy.deepcopy(self.q_net)
        
        
    def gradient_counter(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._gradient_counter += num
        elif pattern == 'set':
            self._gradient_counter = num           
        return self._gradient_counter 

    
    def gradient_timer(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._gradient_timer += num
        elif pattern == 'set':
            self._gradient_timer = num           
        return self._gradient_timer 
    
    
    def reset(self):
        super().reset()
        self.gradient_counter('set', 0)
        self.gradient_timer('set', 0)
        self.loss_running_mean = 0
        self.loss_running_std = 0
    
    
    def set_gradient_scheme(
            self,
            gamma=.99, 
            gradient_freq=1, 
        ):
        # set attributes
        self.batch_size = self.replay.batch_size
        self.gamma = gamma
        assert gradient_freq >= 1
        self.gradient_freq = gradient_freq

    
    def set_optimize_scheme(self, lr=.0001, optimize_freq=1):
        self.lr = lr
        assert optimize_freq >= 1
        self.optimize_freq = optimize_freq
        self.optimizer = self._get_optimizer(
            self.q_net.parameters(), 
            lr=self.lr
            )
        self.optimizer.zero_grad()
        
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        print('______________________________rank [%s] target network updated'
              '______________________________' % self.rank, flush=True)
        
    def backward(self):
        self.gradient_timer('add')
        if self.gradient_timer() % self.gradient_freq != 0: return
        self.gradient_counter('add')
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
            targeted_q_values[non_final_mask] = self._non_final_targeted_q_values(non_final_next)
        except TypeError: print('encountered a case where all next states are None', flush=True)
        # compute the expected Q values
        expected_q_values = (targeted_q_values * self.gamma) + rewards
        # compute loss
        q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1))
        # optimize the model
        self.q_net.zero_grad()
        grad = autograd.grad(q_net_loss, self.q_net.parameters())
        delta_grad = parameters_to_vector(grad)[self.shard_idx]
        self.gradient.add_(delta_grad)
        return self.gradient
    
    
    def zero_grad_(self):
        self.gradient = self.param_vector[self.shard_idx]
    
    
    def optimize(self):
        self.optimize_timer('add')
        if self.optimize_timer() % self.optimize_freq != 0: 
            autograd.backward(self.q_net.parameters(), self.gradient.parameters())
            return
        self.optimize_counter('add')
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        