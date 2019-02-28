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
#         print('param len %s shard len %s' % (self.param_len, self.shard_len), flush=True)

        
        
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
        comm='cpu',
        ):
        super(GorilaDQN_ServerAgent, self).__init__(
            device, 
            q_net, 
            optimizer_func=optimizer_func, 
            comm=comm, 
        )
        
#         self.param_list = [item.clone() for item in self.q_net.parameters()]
        autograd.backward(self.q_net.parameters(), self.q_net.parameters())
        self.grad_list = [item.grad for item in self.q_net.parameters()]
#         for x in self.q_net.parameters():
#             print('grad_list before:', x.grad[0][0][0].tolist(), flush=True)
#             break
        self.q_net.zero_grad()
#         for x in self.q_net.parameters():
#             print('grad_list after:', x.grad[0][0][0].tolist(), flush=True)
#             break
        
        self.save = lambda: None
        
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
        self.master_shard = torch.zeros(self.shard_len, dtype=torch.long, device=self.comm)
        for rank in range(self.world_size):
            if rank == self.master_rank: 
                self.shard_mask[rank] = self.master_shard
                continue
            self.shard_mask[rank] = \
                self.param_perm[(self.param_perm % self.num_clients) == self.rank_to_shard[rank]]
            if len(self.shard_mask[rank]) < self.shard_len:
                self.shard_mask[rank] = torch.cat(
                    (self.shard_mask[rank], torch.randint(self.param_len, (1,)).to(self.comm)))
            assert len(self.shard_mask[rank]) == self.shard_len, 'length error'
#             print('assert %s %s' % (len(self.shard_mask[rank]),  self.shard_len), flush=True)
                
        # send to clients
        dist.scatter(self.master_shard, scatter_list=self.shard_mask, src=self.master_rank)
    
    
    def set_optimize_scheme(self, lr=.0001, optimize_freq=1):
        self.lr = lr
        assert optimize_freq >= 1
        self.optimize_freq = optimize_freq
        self.optimizer = self._get_optimizer(
            self.q_net.parameters(), 
            lr=self.lr
        )
        self.zero_grad_()
        
    
    def set_checkpoint(self, save_freq, save_path):
        self.save_freq = save_freq
        self.save_path = save_path
        def _save():
            if self.optimize_counter() % self.save_freq == 0:
                self.save_pth(self.q_net, self.save_path,
                    filename='q_net.pth', obj_name='q_network')
        self.save = _save
    
    
    def optimize(self, rank, grad_shard):
        self.optimize_timer('add')
#         self.param_vector[self.shard_mask[rank]].add_(grad_shard)
        self.param_vector[self.shard_mask[rank]] = grad_shard
        if self.optimize_timer() % self.optimize_freq != 0: return
#         print('grad_shard norm:', grad_shard.norm(), flush=True)
#         print('param_vector norm:', self.param_vector.norm(), flush=True)
        vector_to_parameters(self.param_vector, self.grad_list)
#         for x in self.q_net.parameters():
#             print('>>>>>>before backward:', x[0][0][0].tolist(), flush=True)
#             break
#         print('>>>>>grad norm<<<<<<<:', grad_shard.norm(), flush=True)
#         for x in self.q_net.parameters():
#             print('param_list norm:', parameters_to_vector(self.param_list).norm(), flush=True)
#             break
#         autograd.backward(self.q_net.parameters(), self.param_list)
#         autograd.backward(self.q_net.parameters(), self.grad_list)
#         for idx, x in enumerate(self.q_net.parameters()):
#             x.grad = self.param_list[idx]
        self.optimize_counter('add')
        self.optimizer.step()
#         print('params norm:', parameters_to_vector(self.q_net.parameters()).norm(), flush=True)
#         for x in self.q_net.parameters():
#             print('after backward<<<<<<<:', x[0][0][0].tolist(), flush=True)
#             break
        self.zero_grad_()
        self.save()

        
    def zero_grad_(self):
        self.param_vector = self.param_vector.zero_()
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
        several=6, 
        comm='cpu',
    ):
        super(GorilaDQN_ClientAgent, self).__init__(
            device, 
            q_net, 
            target_net, 
            loss_func=loss_func, 
            double_dqn=double_dqn, 
            comm=comm, 
        )
        
        self.gradient = None
        self.loss_running_mean = 0.
        self.loss_running_std = 1.
        self.momentum = momentum
        self.several = several
        self.shard = self.rank if self.rank <= self.master_rank else self.rank - 1
        self.shard_mask = torch.zeros(self.shard_len, dtype=torch.long, device=self.comm)
        dist.scatter(self.shard_mask, [], src=self.master_rank)
#         print('rank %s, scattered shard_mask len %s' % (self.rank, len(self.shard_mask)), flush=True)
        
    
    def reset(self):
        super().reset()
        self.loss_running_mean = 0.
        self.loss_running_std = 1.
    
        
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
        
    
    def _update_loss_running_mean(self, loss):
        new_mean = loss.mean().detach()
        self.loss_running_mean = \
            (1 - self.momentum) * self.loss_running_mean + self.momentum * new_mean
        return self.loss_running_mean
    
    
    def _update_loss_running_std(self, loss):
        new_std = loss.std().detach()
        self.loss_running_std = (1 - self.momentum) * self.loss_running_std + self.momentum * new_std
        return self.loss_running_std
    
    
    def _loss_upper_bound(self):
        return self.loss_running_mean + self.several * self.loss_running_std
    
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
#         print('[rank %s] ______________________________ target network updated'
#               '______________________________' % self.rank, flush=True)
        
        
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
#         q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1), reduction='none')
#         self._update_loss_running_mean(q_net_loss)
#         self._update_loss_running_std(q_net_loss)
#         q_net_loss = q_net_loss[q_net_loss <= self._loss_upper_bound()].mean()
        # optimize the model
        grad = autograd.grad(q_net_loss, self.q_net.parameters())
        delta_grad = parameters_to_vector(grad)[self.shard_mask]
        assert self.gradient is not None, 'should call zero_grad_() before backward'
        self.gradient.add_(delta_grad)
        self._record(rewards, q_net_loss, predicted_q_values, expected_q_values, self.gradient_counter)
#         print('rank %s grad len %s' % (self.rank, len(self.gradient)), flush=True)
#         print('grad norm:', self.gradient.norm(), flush=True)
        return self.gradient
    
    
    def zero_grad_(self):
        self.q_net.zero_grad()
        self.gradient = self.param_vector[self.shard_mask].clone()
        
        
        