import os
import torch
import torch.distributed as dist


def get_world_size():
    return int(os.environ['SLURM_NTASKS'])


def get_rank():
    return int(os.environ['SLURM_PROCID'])


def get_jobid():
    return int(os.environ['SLURM_JOBID'])


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)


def slurm_dist(port, backend):
    os.environ['DISTRIBUTED_BACKEND'] = backend

    rank = get_rank()
    world_size = get_world_size()
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0: 
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
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

        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        dist.init_process_group(backend=backend)

    return rank, world_size
