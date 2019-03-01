import os
import threading
from threading import Thread
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pytorl.utils import Setting


def get_master_rank():
    return int(os.environ['MASTER_RANK'])
 
    
class SIG:
    GRAD_PUSH = 0
    PARAM_REQ = 1
    QUERY = 2
    GRAD = 3
    PARAM = 4
    

    
class _Messenger(Thread):
    def __init__(self, device):
        super(_Messenger, self).__init__()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.device = device

    
    def isend(self, overhead, payload=None, dst=0, tag=0, comm='cpu'):
        _overhead_msg, _payload_msg = overhead, payload
        if not isinstance(_overhead_msg, torch.Tensor): 
            if hasattr(_overhead_msg, '__iter__'):
                _overhead_msg = torch.tensor(overhead, dtype=torch.float32)
            else:
                _overhead_msg = torch.tensor([overhead], dtype=torch.float32)
        msg = _overhead_msg.to(comm)
        if _payload_msg is not None:
            if not isinstance(_payload_msg, torch.Tensor):
                try:
                    _payload_msg = parameters_to_vector(payload).detach()
                except:
                    raise TypeError('unrecognized payload type, not a vector tensor nor an iterator')
            _payload_msg = _payload_msg.to(comm)
            msg = torch.cat((msg, _payload_msg))
        return dist.isend(msg, dst, tag=tag)
        

    def recv(self, overhead_len, payload_len=0, src=None, tag=0, comm='cpu'):
        msg = torch.zeros(overhead_len, dtype=torch.float32, device=comm)
        if payload_len > 0: 
            _payload_msg = torch.zeros(payload_len, dtype=torch.float32, device=comm)
            msg = torch.cat((msg, _payload_msg))
        dist.recv(msg, src, tag=tag)
        if payload_len > 0:
            _overhead_msg, _payload_msg = msg[:overhead_len], msg[overhead_len:].to(self.device)
            _overhead_msg = list(map(int, _overhead_msg))
            return _overhead_msg, _payload_msg
        
        return list(map(int, msg))
    
    
    def run(self):
        raise NotImplementedError('cannot run base _Messenger class')
    
    

class ParamServer(_Messenger):
    def __init__(self, device, thread, lock):
        super(ParamServer, self).__init__(device)
        self.master_rank = get_master_rank()
        self.thread = thread
        self.lock = lock
    
    
    @Setting.only_once
    def set_listen(self, recv_info_len, global_timesteps_counter):
        """msg format: [rank, shard_len, local_timesteps, signal]"""
        self.recv_info_len = recv_info_len
        self.counter = global_timesteps_counter
    
    
    @Setting.only_once
    def set_param_update(self, model, optim_handler):
        self.model = model
        self.optim_handler = optim_handler
        self.param_vector = parameters_to_vector(model.parameters()).detach()
    
    
    def _recv_info(self):
        return self.recv(self.recv_info_len, tag=SIG.QUERY)
    
    
    def _recv_shard(self, src, shard_len):
        return self.recv(self.recv_info_len, shard_len, src=src, tag=SIG.GRAD)
    
    
    def _isend_param(self, dst):
        return self.isend(
            [self.counter(), self.thread], self.model.parameters(), dst=dst, tag=SIG.PARAM)
    
    
    def listen(self):
        sender, shard_len, local_time, signal = self._recv_info()
        if signal == SIG.GRAD_PUSH: 
            _, grad_shard = self._recv_shard(sender, shard_len)
            with self.lock: self.optim_handler(sender, grad_shard)
            return
        self._isend_param(sender).wait()
    
    
    def run(self):
        while True:
            self.listen()
  
    
    
class ParamClient(_Messenger):
    def __init__(self, device):
        super(ParamClient, self).__init__(device)
        self.master_rank = get_master_rank()
        self.overhead = None
    
    
    @Setting.only_once
    def set_recv(self, recv_info_len):
        """msg format: [global_timesteps, server_thread_num]"""
        self.recv_info_len = recv_info_len
    
    
    @Setting.only_once
    def set_info(self, shard_idx, local_timesteps_counter):
        self.shard_idx = shard_idx
        self.counter = local_timesteps_counter
    
    
    @Setting.only_once
    def set_param_update(self, model):
        self.model = model
        self.param_vector = parameters_to_vector(model.parameters()).detach()
        self.model_len = len(self.param_vector)
    
    
    def _isend_info(self, signal):
        self.overhead = [self.rank, self.shard_idx, self.counter(), signal]
        return self.isend(self.overhead, dst=self.master_rank, tag=SIG.QUERY)
    
    
    def recv_param(self):
        self._isend_info(SIG.PARAM_REQ).wait()
        return self.recv(self.recv_info_len, self.model_len, src=self.master_rank, tag=SIG.PARAM)
    
    
    def isend_shard(self, shard_data):
        self._isend_info(SIG.GRAD_PUSH).wait()
        self.isend(self.overhead, shard_data, dst=self.master_rank, tag=SIG.GRAD).wait()
    
    
    def run(self):
        raise RuntimeError('cannot run parameter client')
    

    