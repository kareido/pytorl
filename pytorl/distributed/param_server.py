import os
import threading
from threading import Thread
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def get_master_rank():
    return int(os.environ['MASTER_RANK'])
 
    
class SIG:
    GRAD_PUSH = 0
    PARAM_REQ = 1
    QUERY = 2
    GRAD = 3
    PARAM = 4
    

    
class _Messenger(Thread):
    def __init__(self):
        super(_Messenger, self).__init__()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def isend(self, overhead, payload=None, dst=0, tag=0, comm='cpu'):
        _overhead_msg, _payload_msg = overhead, payload
        if not isinstance(_overhead_msg, torch.Tensor): 
            if hasattr(_overhead_msg, '__iter__'):
                _overhead_msg = torch.tensor(overhead, dtype=torch.float32)
            else:
                _overhead_msg = torch.tensor([overhead], dtype=torch.float32)
        msg = _overhead_msg.to(comm)
        if _payload_msg is not None and not isinstance(_payload_msg, torch.Tensor):
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
            msg = torch.cat((_payload_msg, msg))
        dist.recv(msg, src, tag=tag)
        if payload_len > 0:
            _overhead_msg, _payload_msg = msg[:overhead_len], msg[overhead_len:].to(self.device)
            _overhead_msg = list(map(int, _overhead_msg))
            return _overhead_msg, _payload_msg
        
        return list(map(int, msg))
    
    
    def run(self):
        raise NotImplementedError('cannot run base _Messenger class')
    
    

class ParamServer(_Messenger):
    def __init__(self, thread, lock):
        super(ParamServer, self).__init__()
        self.master_rank = get_master_rank()
        self.thread = thread
        self.lock = lock
    
    
    """msg format: [rank, shard, local_timesteps, signal]"""
    def set_listen(self, recv_info_len, global_timesteps_counter):
        self.recv_info_len = recv_info_len
        self.counter = global_timesteps_counter
    
    
    def set_param_update(self, model, optim_handler):
        self.model = model
        self.optim_handler = optim_handler
        self.param_vector = parameters_to_vector(model.parameters()).detach()
        self.shard_len = (len(self.param_vector) + self.world_size - 1) // (self.world_size - 1)
    
    
    def _recv_info(self):
        return self.recv(self.recv_info_len, tag=SIG.QUERY)
    
    
    def _recv_shard(self, src):
        return self.recv(self.recv_info_len, self.shard_len, src=src, tag=SIG.GRAD)
    
    
    def _isend_param(self, dst):
        return self.isend(
            [self.counter(), self.thread], self.model.parameters(), dst=dst, tag=SIG.PARAM)
    
    
    def listen(self):
        sender, shard, local_time, signal = self._recv_info()
        if signal == SIG.GRAD_PUSH: 
            _, grad_shard = self._recv_shard(sender)
#             print('master server updating q network using grad from rank [%s]' % sender, flush=True)
            with self.lock: self.optim_handler(shard, grad_shard)
            return
        self._isend_param(sender).wait()
    
    
    def run(self):
        while True:
            self.listen()
  
    
    
class ParamClient(_Messenger):
    def __init__(self):
        super(ParamClient, self).__init__()
        self.master_rank = get_master_rank()
        self.overhead = None
    
    """msg format: [global_timesteps, server_thread_num]"""
    def set_recv(self, recv_info_len):
        self.recv_info_len = recv_info_len
    
    
    def set_info(self, shard_idx, local_timesteps_counter):
        self.shard_idx = shard_idx
        self.counter = local_timesteps_counter
    
    
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
    
