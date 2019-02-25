import os
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def get_master_rank():
    return int(os.environ['MASTER_RANK'])


class Messenger:
    def __init__(self, rank, master_rank):
        self.rank = rank
        self.master_rank = master_rank
        self._push_timer = 0
        self._push_counter = 0
        self.broadcast_params_finished = lambda: True
    
    def push_counter(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._push_counter += num
        elif pattern == 'set':
            self._push_counter = num           
        return self._push_counter 

    
    def push_timer(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._push_timer += num
        elif pattern == 'set':
            self._push_timer = num           
        return self._push_timer 
    
    
    def set_master_rank(self, shard_idx, padded_len, updates_counter, worker_list):
        assert self.rank == self.master_rank
        self.shard_idx = shard_idx
        self.updates_counter = updates_counter
        self.padded_len = padded_len
        self.worker_list = worker_list
        self.push_params_finished = [lambda: True] * (len(worker_list) + 1)
        
        
    def set_learner_rank(self, shard, push_freq):
        self.shard = shard
        self.push_freq = push_freq
    
    
    def push_grad_shard(self, grads):
        self.push_timer('add')
        if self.push_timer() % self.push_freq != 0: return False
        self.push_counter('add')
        assert self.rank != self.master_rank, 'illegal pushing'
        tensor = grads.detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        if dist.get_backend() == 'gloo': tensor = tensor.cpu()
        comm_device = 'cuda' if tensor.is_cuda else 'cpu'
        sender = torch.tensor([self.rank], dtype=torch.float32, 
                              device=comm_device, requires_grad=False)
        shard = torch.tensor([self.shard], dtype=torch.float32, 
                             device=comm_device, requires_grad=False)
        msg = torch.cat((tensor, sender, shard))
        dist.send(msg, self.master_rank)
        return True
    
    
    def pull_grad_shard(self, model):
        assert self.rank == self.master_rank, 'illegal pulling'
        tensor = parameters_to_vector(model.parameters()).zero_().detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        comm_device = 'cpu' if dist.get_backend() == 'gloo' else device
        masked_tensor = torch.zeros(self.padded_len).to(comm_device)
        sender = torch.tensor([0.], dtype=torch.float32, device=comm_device, requires_grad=False)
        shard = torch.tensor([0.], dtype=torch.float32, device=comm_device, requires_grad=False)
        msg = torch.cat((masked_tensor, sender, shard))
        dist.recv(msg)
        masked_tensor, sender, shard = msg[:-2].to(device), msg[-2].to(device), msg[-1].to(device)
        sender, shard = int(sender), int(shard)
        tensor[self.shard_idx[shard]] = masked_tensor
        vector_to_parameters(tensor, model.parameters())
        return sender, shard
    
    
    def push_params(self, params, forced=False):
        assert self.rank == self.master_rank, 'illegal broadcasting'
        tensor = parameters_to_vector(params).detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        if dist.get_backend() == 'gloo': tensor = tensor.cpu()
        comm_device = 'cuda' if tensor.is_cuda else 'cpu'
        updates = torch.tensor([self.updates_counter()], dtype=torch.float32, 
                           device=comm_device, requires_grad=False)
        msg = torch.cat((tensor, updates))
        for worker in self.worker_list:
            if not self.push_params_finished[worker]() and not forced: 
                if not self.push_params_prev_finished[worker](): continue
            print('______________________________master rank [%s] send network parameters to rank [%s]'
              '______________________________' % (self.rank, worker), flush=True)
            handler = dist.isend(msg, worker)
            self.push_params_finished[worker] = handler.is_completed
        return self.push_params_finished

    
    def pull_params(self, params_receiver):
        assert self.rank != self.master_rank, 'illegal pulling'
        tensor = parameters_to_vector(params_receiver).detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        if dist.get_backend() == 'gloo': tensor = tensor.cpu()
        comm_device = 'cuda' if tensor.is_cuda else 'cpu'
        updates = torch.tensor([0.], dtype=torch.float32, device=comm_device, requires_grad=False)
        msg = torch.cat((tensor, updates))
        handler = dist.recv(msg, self.master_rank)
#         handler.wait()
#         dist.recv(msg, self.master_rank)
#         print('rank [%s] received' % self.rank, flush=True)
        tensor, updates = msg[:-1].to(device), msg[-1].to(device)
        vector_to_parameters(tensor, params_receiver)
        return int(updates)
    
    
    def broadcast_params(self, params, forced=False):
        assert self.rank == self.master_rank, 'illegal broadcasting'
        if not self.broadcast_params_finished() and not forced: return
        print('______________________________master rank [%s] broadcasting network parameters'
              '______________________________' % self.rank, flush=True)
        tensor = parameters_to_vector(params).detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        if dist.get_backend() == 'gloo': tensor = tensor.cpu()
        comm_device = 'cuda' if tensor.is_cuda else 'cpu'
        updates = torch.tensor([self.updates_counter()], dtype=torch.float32, 
                               device=comm_device, requires_grad=False)
        msg = torch.cat((tensor, updates))
        handler = dist.broadcast(msg, self.master_rank, async_op=True)
        self.broadcast_params_finished = handler.is_completed
        return handler
   

    def pull_broadcast_params(self, params_receiver):
        assert self.rank != self.master_rank, 'illegal pulling'
        tensor = parameters_to_vector(params_receiver).detach()
        device = 'cuda' if tensor.is_cuda else 'cpu'
        if dist.get_backend() == 'gloo': tensor = tensor.cpu()
        comm_device = 'cuda' if tensor.is_cuda else 'cpu'
        updates = torch.tensor([0.], dtype=torch.float32, device=comm_device, requires_grad=False)
        msg = torch.cat((tensor, updates))
        dist.broadcast(msg, self.master_rank, async_op=False)
        tensor, updates = msg[:-1].to(device), msg[-1].to(device)
        vector_to_parameters(tensor, params_receiver)
        return int(updates)
    
    