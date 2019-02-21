import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from pytorl.lib import PrioritizedReplay
from ._base_agent import Agent


class DQN_Agent(Agent):
    def __init__(self, 
         device, 
         q_net, 
         target_net=None, 
         loss_func=None, 
         optimizer_func=None,  
         replay=None, 
        ):
        self.device = device
        self.q_net = q_net
        self.target_net = target_net
        self.loss = loss_func
        self._get_optimizer = optimizer_func
        self.replay = replay
        # attributes for optimization
        self.batch_size = None
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
        self.batch_size = self.replay.batch_size
        self.lr = lr
        self.gamma = gamma
        self.optimize_freq = optimize_freq
        self.update_target_freq = update_target_freq
        self.optimizer = self._get_optimizer(
            self.q_net.parameters(), 
            lr=self.lr
        )
    
    
    def reset(self):
        if self.replay: self.replay.clear()
        if self.target_net: self.set_device()
        self.optimize_counter('set', 0)
        self.optimize_timer('set', 0)
        for name, params in self.q_net.named_parameters():
            if 'bias' in name:
                # to avoid 'Fan in and fan out can not be computed for tensor with fewer 
                # than 2 dimensions' problem
                nn.init.zeros_(params)
            else:
                nn.init.kaiming_normal_(params)
        if self.target_net: self.update_target()
        self.q_net.train(True)
        if self.target_net: self.target_net.train(False)
    
    
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
    
    
    """ should check if non_final_next is None when call this method"""
    def _non_final_targeted_q_values(self, non_final_next):
        return self.target_net(non_final_next).max(1)[0].detach()
    
    
    def _record(self, rewards, q_net_loss, predicted_q_values, expected_q_values):
        if self._tensorboard is not None:
            reward_mean = rewards.mean().item()
            predicted_q_values_mean = predicted_q_values.mean().item()
            expected_q_values_mean = expected_q_values.mean().item()

            self._tensorboard.add_scalar('timestep/replay_reward-mean', 
                                   reward_mean, self.optimize_counter())
            self._tensorboard.add_scalar('timestep/loss', q_net_loss, self.optimize_counter())
            self._tensorboard.add_scalar('timestep/predicted_q_values-mean', 
                                   predicted_q_values_mean, self.optimize_counter())
            self._tensorboard.add_scalar('timestep/expected_q_values-mean', 
                                   expected_q_values_mean, self.optimize_counter())    
    
    
    def optimize(self):
        self.optimize_timer('add')
        if self.optimize_timer() % self.optimize_freq != 0: return
        self.optimize_counter('add')
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
        self.optimizer.zero_grad()
        q_net_loss.backward()
        clip_grad_value_(self.q_net.parameters(), 1)
        self.optimizer.step()
        # update target network
        if self.optimize_counter() % self.update_target_freq == 0:
            self.update_target()
        # tensorboard recording
        self._record(rewards, q_net_loss, predicted_q_values, expected_q_values)
        
        
            
class DoubleDQN_Agent(DQN_Agent):
    def __init__(self, 
         device, 
         q_net, 
         target_net=None, 
         loss_func=None, 
         optimizer_func=None,  
         replay=None, 
        ):
        super(DoubleDQN_Agent, self).__init__(
            device, q_net, 
            target_net=target_net, 
            loss_func=loss_func, 
            optimizer_func=optimizer_func,  
            replay=replay, 
        )
        
    
    """ should check if non_final_next is None when call this method"""
    def _non_final_targeted_q_values(self, non_final_next):
        # must view it to match the shape
        next_actions = self.q_net(non_final_next).max(1)[1].view(-1, 1)
        # must squeeze it to make it a batch of scalar values
        return self.target_net(non_final_next).gather(1, next_actions).squeeze()
    
    
    
class PrioritizedDQN_Agent(DQN_Agent):
    def __init__(self, 
         device, 
         q_net, 
         target_net=None, 
         loss_func=None, 
         optimizer_func=None, 
         replay=None, 
         double_dqn=True, 
        ):
        super(PrioritizedDQN_Agent, self).__init__(
            device, q_net, 
            target_net=target_net, 
            loss_func=loss_func, 
            optimizer_func=optimizer_func,  
        )
        self.replay = replay
        if double_dqn:
            self._non_final_targeted_q_values = self._double_dqn_q_values
        else:
            self._non_final_targeted_q_values = self._natural_dqn_q_values
        
    
    def set_prioritized_replay(self, capacity=None, batch_size=32, 
                               init_size=None, alpha=1, beta_func=lambda: 1, eps=1e-6):
        self.replay = PrioritizedReplay(
            capacity=capacity, 
            batch_size=batch_size, 
            init_size=init_size, 
            alpha=alpha,
            beta_func=beta_func, 
            eps=eps, 
        )
        
    
    """ should check if non_final_next is None when call this method"""
    def _double_dqn_q_values(self, non_final_next):
        # must view it to match the shape
        next_actions = self.q_net(non_final_next).max(1)[1].view(-1, 1)
        # must squeeze it to make it a batch of scalar values
        return self.target_net(non_final_next).gather(1, next_actions).squeeze()
    
    
    """ should check if non_final_next is None when call this method"""    
    def _natural_dqn_q_values(self, non_final_next):
        # must view it to match the shape
        next_actions = self.q_net(non_final_next).max(1)[1].view(-1, 1)
        # must squeeze it to make it a batch of scalar values
        return self.target_net(non_final_next).gather(1, next_actions).squeeze()
       
    
    def optimize(self):
        self.optimize_timer('add')
        if self.optimize_timer() % self.optimize_freq != 0: return
        self.optimize_counter('add')
        sample_exp = self.replay.sample()
        batch = self.replay.form_obj(*zip(*sample_exp))
        curr_states = torch.cat(batch.curr_state).to(self.device)
        actions = torch.tensor(batch.action).to(self.device).view(-1, 1)
        rewards = torch.tensor(batch.reward).to(self.device)
        weights = torch.tensor(batch.weight).to(self.device)
        indices = torch.tensor(batch.index).to(self.device)
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
        # compute temporal difference error
        td_error = predicted_q_values - expected_q_values.unsqueeze(1)
        new_priorities = (torch.abs(td_error.squeeze()) + self.replay.eps).tolist()
        self.replay.update_priorities(indices, new_priorities)
        # compute loss
        q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1), reduction='none')
        q_net_loss = torch.dot(weights, q_net_loss.squeeze())
        # optimize the model
        self.optimizer.zero_grad()
        q_net_loss.backward()
        clip_grad_value_(self.q_net.parameters(), 1)
        self.optimizer.step()
        # update target network
        if self.optimize_counter() % self.update_target_freq == 0:
            self.update_target()
        # tensorboard recording
        self._record(rewards, q_net_loss, predicted_q_values, expected_q_values)
    
    
    
    