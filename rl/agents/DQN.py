import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from ._base_agent import Agent
import rl.networks as network


class DQN_Agent(Agent):
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
        self.loss = loss_func
        self._get_optimizer = optimizer_func
        self._lr = lr
        self.replay = replay
        self.optimizer = self._get_optimizer(
                            self.q_net.parameters(), 
                            lr=self._lr)
        self._optimize = None
    
    
    def reset(self):
        self.replay.clear()
        self.set_device()
        for name, params in self.q_net.named_parameters():
            if 'bias' in name:
                # to avoid 'Fan in and fan out can not be computed 
                # for tensor with fewer than 2 dimensions' problem
                nn.init.zeros_(params)
            else:
                nn.init.kaiming_normal_(params)
        self.update_target()
        self.q_net.train(True)
        self.target_net.train(False)
        self.optimizer = self._get_optimizer(
                            self.q_net.parameters(), 
                            lr=self._lr)
        self._optimize = None
    
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    
    def set_device(self):
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        
    
    def next_action(self, thres, rand_act, curr_input):
        sample_val = random.random()
        if sample_val > thres:
            with torch.no_grad():
                curr_q_val = self.q_net(curr_input.to(self.device))
                return curr_q_val.argmax(1).view(1, 1).to('cpu')
        else:
            # guarantee the return value is legal
            return torch.tensor(rand_act).view(1, 1).to('cpu')
        
        
    """
    add tensorboard and env to visualize training
    """
    def set_optimize_func(self, batch_size, gamma, min_replay=None, 
                          learn_freq=1, tensorboard=None, counter=None):
        assert type(learn_freq) == int and learn_freq >= 1
        if tensorboard is not None and learn_freq >= 2: assert counter
        if min_replay is None: min_replay = batch_size
        def _optimize():
            # update condition
            if len(self.replay) < min_replay:
                return
            if learn_freq >= 2 and counter() % learn_freq != 0:
                return
            sample_exp = self.replay.sample(batch_size)
            batch = self.replay.form_obj(*zip(*sample_exp))
            curr_states = torch.cat(batch.curr_state).to(self.device)
            actions = torch.cat(batch.action).to(self.device)
            rewards = torch.tensor(batch.reward).to(self.device)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                          device=self.device, dtype=torch.uint8)
#             np_next_state = np.array(batch.next_state)
#             np_non_final_mask = (np_next_state == None).astype(np.uint8)
#             np_final_mask = 1 - np_non_final_mask
#             non_final_mask = torch.tensor(np_non_final_mask, device=self.device)
#             final_mask = torch.tensor(np_final_mask, device=self.device)
            non_final_next_states = torch.cat(
                                    [s for s in batch.next_state if s is not None]).to(self.device)
#             final_state = curr_states[np_final_mask]
            
            predicted_q_values = self.q_net(curr_states).gather(1, actions)
            targeted_q_values = torch.zeros(rewards.shape[0], device=self.device)
            # compute Q values via stationary target net
            # this 'try' is to avoid the situation when all next states are None
            try:
                targeted_q_values[non_final_mask] = self.target_net(
                                                        non_final_next_states).max(1)[0].detach()
            except TypeError: pass
            # compute the expected Q values
            expected_q_values = (targeted_q_values * gamma) + rewards
#             print((predicted_q_values - expected_q_values.unsqueeze(1)).sum(), flush=True)
            # compute loss
            q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1))
#             print('shapes:', predicted_q_values.shape, targeted_q_values.shape, expected_q_values.shape, q_net_loss.shape, flush=True)
            # optimize the model
            self.optimizer.zero_grad()
            q_net_loss.backward()
            clip_grad_value_(self.q_net.parameters(), 1)
            self.optimizer.step()
            # tensorboard recording
            if tensorboard is not None:
                reward_mean = rewards.mean().item()
                predicted_q_values_mean = predicted_q_values.mean().item()
                targeted_q_values_mean = targeted_q_values.mean().item()
                expected_q_values_mean = expected_q_values.mean().item()
                
                tensorboard.add_scalar('step_replay/reward-mean', 
                                       reward_mean, counter())
                tensorboard.add_scalar('step/loss', q_net_loss, counter())
                tensorboard.add_scalar('step/predicted_q_values-mean', 
                                       predicted_q_values_mean, counter())
                tensorboard.add_scalar('step/targeted_q_values-mean', 
                                       targeted_q_values_mean, counter())
                tensorboard.add_scalar('step/expected_q_values-mean', 
                                       expected_q_values_mean, counter())
                
        self._optimize = _optimize
    
        
    def optimize(self):
        if not self._optimize:
            raise ValueError('should call set_optimize_func first')
        self._optimize()
        
        
        
        
    
        