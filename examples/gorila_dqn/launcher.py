import os
import pytorl.distributed as rl_dist
import torch.distributed as dist
from client import param_client_proc
from server import param_server_proc


os.environ.setdefault('run_name', 'default')


def main():
    rank, world_size, master_rank, worker_list = rl_dist.slurm_param_server_arch(port=23300)
    worker_group = dist.new_group(ranks=worker_list)
    
    if rank == master_rank:
        print('master service running at rank [%s]' % rank, flush=True)
        param_server_proc(master_rank, worker_list)
    else:
        param_client_proc(master_rank, worker_group)
    

if __name__ == '__main__':
    main()
