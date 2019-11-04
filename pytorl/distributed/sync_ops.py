import torch
import torch.distributed as dist


def all_reduce_mean(tensor_list, group=None):
    handler_list = []
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if dist.get_world_size() == 1: return
    for tensor in tensor_list:
        _allreduce(tensor)
        handler_list.append(handler)
        tensor.div_(dist.get_world_size())


def all_reduce_sum(tensor_list, group=None):
    handler_list = []
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.SUM, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if dist.get_world_size() == 1: return
    for tensor in tensor_list:
         _allreduce(tensor)


def all_reduce_max(tensor_list, group=None):
    handler_list = []
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if dist.get_world_size() == 1: return
    for tensor in tensor_list:
        _allreduce(tensor)


def all_reduce_min(tensor_list, group=None):
    handler_list = []
    if group is None:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX)
    else:
        _allreduce = lambda tensor: dist.all_reduce(tensor, dist.ReduceOp.MAX, group)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if dist.get_world_size() == 1: return
    for tensor in tensor_list:
        tensor.neg_()
        _allreduce(tensor)
        tensor.neg_()


def broadcast(tensor_list, src, group=None):
    if group is None:
        _broadcast = lambda tensor: dist.broadcast(tensor, src)
    else:
        _broadcast = lambda tensor: dist.broadcast(tensor, src, group)
    if dist.get_world_size() == 1: return
    for tensor in tensor_list:
        _broadcast(tensor)


def barrier(group=None):
    if dist.get_world_size() == 1: return
    if group is None:
        _barrier = dist.barrier
    else:
        _barrier = lambda: dist.barrier(group)
    _barrier()



