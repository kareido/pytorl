import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from ._Base import BaseAgent
import rl.networks as network


class DQN_Agent(BaseAgent):
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
                nn.init.xavier_normal_(params)
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
        
    
    def next_action(self, thres, rand_act, curr_input=None):
        sample_val = random.random()
        if sample_val > thres and curr_input is not None:
            with torch.no_grad():
                curr_q_val = self.q_net(curr_input.to(self.device))
                return curr_q_val.argmax(1).view(1, 1).to('cpu')
        else:
            # guarantee the return value is legal
            return rand_act.view(1, 1).to('cpu')
        
        
    """
    add tensorboard and env to visualize training
    """
    def set_optimize_func(self, batch_size, gamma, tensorboard=None, env=None):
        if tensorboard is not None: assert env
        def _optimize():
            if len(self.replay) < batch_size:
                return
            sample_exp = self.replay.sample(batch_size)
            batch = self.replay.form_obj(*zip(*sample_exp))
            curr_state_batch = torch.cat(batch.curr_state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.tensor(batch.reward).to(self.device)
            predicted_q_values = self.q_net(curr_state_batch).gather(1, action_batch)
            # compute Q values via stationary target net
            targeted_q_values = self.target_net(curr_state_batch).max(1)[0].detach()
            # compute the expected Q values
            expected_q_values = (targeted_q_values * gamma) + reward_batch

            # compute Huber loss
            q_net_loss = self.loss(predicted_q_values, expected_q_values.unsqueeze(1))
            # optimize the model
            self.optimizer.zero_grad()
            q_net_loss.backward()
            clip_grad_value_(self.q_net.parameters(), 1)
            self.optimizer.step()
            # tensorboard recording
            if tensorboard is not None:
                reward_mean = reward_batch.mean().item()
                predicted_q_values_mean = predicted_q_values.mean().item()
                targeted_q_values_mean = targeted_q_values.mean().item()
                expected_q_values_mean = expected_q_values.mean().item()
                
                tensorboard.add_scalar('step_replay/reward-mean', reward_mean, 
                                       env.total_step_count)
                tensorboard.add_scalar('step/loss', q_net_loss, env.total_step_count)
                tensorboard.add_scalar('step/predicted_q_values-mean', 
                                       predicted_q_values_mean, env.total_step_count)
                tensorboard.add_scalar('step/targeted_q_values-mean', 
                                       targeted_q_values_mean, env.total_step_count)
                tensorboard.add_scalar('step/expected_q_values-mean', 
                                       expected_q_values_mean, env.total_step_count)
                
        self._optimize = _optimize
    
        
    def optimize(self):
        if not self._optimize:
            raise ValueError('should call set_optimize_func first')
        self._optimize()
        
        
        
        
    
        