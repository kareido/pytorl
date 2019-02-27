import copy
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.distributed as dist
from pytorl.distributed import get_master_rank
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .DQN import PrioritizedDQN_Agent


class _GorilaDQN_BaseAgent(PrioritizedDQN_Agent):
    def __init__(
        self, 
        device, 
        q_net,
        target_net=None, 
        loss_func=None, 
        optimizer_func=None, 
        double_dqn=True, 
        comm='cpu'
    ):
        super(_GorilaDQN_BaseAgent, self).__init__(
             device, 
             q_net, 
             target_net=target_net, 
             loss_func=loss_func, 
             optimizer_func=optimizer_func, 
             double_dqn=double_dqn, 
        )
        self.replay = None
        self.comm = comm
        if double_dqn:
            self._non_final_targeted_q_values = self._double_dqn_q_values
        else:
            self._non_final_targeted_q_values = self._natural_dqn_q_values
        
        # distributed env
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.master_rank = get_master_rank()
        self.num_clients = self.world_size - 1
        
        self._gradient_counter = 0
        self._gradient_timer = 0
        self.param_vector = parameters_to_vector(self.q_net.parameters()).zero_().detach()
        self.param_len = len(self.param_vector)
        self.shard_len = (self.param_len + self.num_clients - 1) // self.num_clients

        
        
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
        
        
        
class GorilaDQN_ServerAgent(_GorilaDQN_BaseAgent):
    def __init__(
        self, 
        device, 
        q_net, 
        optimizer_func=None, 
        ):
        super(GorilaDQN_ServerAgent, self).__init__(
            device, 
            comm, 
            q_net, 
            optimizer_func=optimizer_func, 
        )
        
        # get rank to shard mapping
        self.rank_to_shard = [None] * self.world_size
        for idx in range(self.world_size):
            if idx < self.master_rank: 
                self.rank_to_shard[idx] = idx
            else: 
                self.rank_to_shard[idx] = idx - 1
        self.rank_to_shard[self.master_rank] = -1
        
        # setup random shards
        self.param_perm = torch.randperm(self.param_len).to(self.comm)    
        self.shard_mask = [None] * self.world_size
        self.master_shard = torch.zeros(1).to(self.comm)
        for rank in range(self.world_size):
            if rank == self.master_rank: 
                self.shard_mask[rank] = self.master_shard
                continue
            self.shard_mask[rank] = self.param_perm[(
                self.param_perm % self.num_clients) == self.rank_to_shard(rank)]
            if len(self.shard_mask[rank]) < self.shard_len:
                self.shard_mask[rank] = torch.cat(
                    (self.shard_mask[rank], torch.randint(self.param_len, (1,)).to(self.comm)))
                assert len(self.shard_mask[idx]) == self.shard_len, 'length error'
                
        # send to clients
        dist.scatter(self.master_shard, scatter_list=self.shard_mask)
    
    
    def set_optimize_scheme(self, lr=.0001, optimize_freq=1):
        self.lr = lr
        assert optimize_freq >= 1
        self.optimize_freq = optimize_freq
        self.optimizer = self._get_optimizer(
            self.q_net.parameters(), 
            lr=self.lr
        )
        self.optimizer.zero_grad()
    
    
    def optimize(self):
        self.optimize_timer('add')
        if self.optimize_timer() % self.optimize_freq != 0: 
            autograd.backward(self.q_net.parameters(), self.gradient.parameters())
            return
        self.optimize_counter('add')
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    
                    
class GorilaDQN_ClientAgent(_GorilaDQN_BaseAgent):
    def __init__(
        self, 
        device, 
        q_net,
        target_net, 
        loss_func, 
        double_dqn=True, 
        momentum=0.1, 
        timeout=0.1,
    ):
        super(GorilaDQN_ClientAgent, self).__init__(
             device, 
             q_net, 
             target_net=target_net, 
             loss_func=loss_func, 
             double_dqn=double_dqn, 
        )
        
        self.loss_running_mean = 0
        self.loss_running_std = 0
        self.momentum = momentum
        self.timeout= timeout
        self.shard = self.rank if self.rank <= self.master_rank else self.rank - 1
        self.shard_mask = torch.zeros(self.shard_len).to(self.comm)
        dist.scatter(self.shard_mask, src=self.master_rank)
        
    
    def reset(self):
        super().reset()
        self.loss_running_mean = 0
        self.loss_running_std = 0
    
        
    def set_gradient_scheme(
            self,
            shard
            gamma=.99, 
            gradient_freq=1, 
        ):
        # set attributes
        self.batch_size = self.replay.batch_size
        self.gamma = gamma
        assert gradient_freq >= 1
        self.gradient_freq = gradient_freq
        
        
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        print('[rank %s] ______________________________ target network updated'
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
        
        
        