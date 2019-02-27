import os
import torch
import torch.distributed as dist
from .slurm_env import *


def _get_slurm_addr():
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    return addr



def slurm_data_parallel_arch(port=23032, backend='nccl'):
    os.environ['DISTRIBUTED_BACKEND'] = backend

    rank = get_rank()
    world_size = get_world_size()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = _get_slurm_addr()
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        dist.init_process_group(backend=backend)

    return rank, world_size


def data_parallel_arch(port=23030, backend='nccl'):
    """
    a DIY way to start data parallel distributed arch
    """
    os.environ['DISTRIBUTED_BACKEND'] = backend
    rank, world_size = get_rank(), get_world_size()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)
    
    if world_size == 1:
        rank, world_size = 0, 1
    
    else:
        checklist = {'MASTER_PORT', 'MASTER_ADDR', 'WORLD_SIZE', 'RANK', 'MASTER_RANK'}
        for env_var in checklist:
            assert env_var in os.environ, 'error: %s not set yet' % env_var
        dist.init_process_group(backend=backend)

    return rank, world_size, master_rank, worker_list


def slurm_param_server_arch(port=23029, backend='gloo', master_rank=0):
    """
    this function inits the parameter server architecture distributed environment, which helps the 
    asynchronized training in the context of reinforcement learning
    
    master_rank serves as the parameter server (master process, and the others are
    training processes (slaves) with the gpu mapping worker[i] -> gpu[i] % (num of gpus)
    
    [!]NOTE: haven't tested under mpi or nccl backend
    """
    
    os.environ['DISTRIBUTED_BACKEND'] = backend
    rank, world_size = get_rank(), get_world_size()
    assert world_size > 1, 'parameter server arch requires multiple processes'
    assert master_rank < world_size, 'invalid master_rank: %s' % master_rank
    worker_list = list(range(world_size))
    del worker_list[master_rank]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = _get_slurm_addr()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['MASTER_RANK'] = str(master_rank)
        
    dist.init_process_group(backend=backend)

    return rank, world_size, master_rank, worker_list



def param_server_arch(port=23028, backend='gloo', master_rank=0):
    """
    a DIY way to start parameter server distributed arch
    """
    os.environ['DISTRIBUTED_BACKEND'] = backend
    rank, world_size = get_rank(), get_world_size()
    assert world_size > 1, 'parameter server arch requires multiple processes'
    assert master_rank < world_size, 'invalid master_rank: %s' % master_rank
    worker_list = list(range(world_size))
    del worker_list[master_rank]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    checklist = {'MASTER_PORT', 'MASTER_ADDR', 'WORLD_SIZE', 'RANK', 'MASTER_RANK'}
    for env_var in checklist:
        assert env_var in os.environ, 'error: %s not set yet' % env_var
        
    dist.init_process_group(backend=backend)

    return rank, world_size, master_rank, worker_list


