import os
import random
import threading
from threading import Lock
import time
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from pytorl.agents import GorilaDQN_ServerAgent
import pytorl.distributed as rl_dist
from pytorl.envs import make_atari_env
import pytorl.lib as lib
from pytorl.networks import Dueling_DQN, Q_Network
import pytorl.utils as utils


def param_server_proc(master_rank, worker_list):
    ################################################################
    # DEVICE
    rank, world_size = dist.get_rank(), dist.get_world_size()
    master_rank = rl_dist.get_master_rank()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('RANK: [%s], current device: [%s]' % (rank, device), flush=True)

    ################################################################
    # CONFIG & SETTINGS
    cfg_reader = utils.ConfigReader(default='run_project/config.yaml')
    config = cfg_reader.get_config()
    seed, frames_stack = config.seed, config.solver.frames_stack
    save_freq, save_path = config.record.save_freq, config.record.save_path
    num_servers, shard_factor = config.server.num_threads, config.server.shard_factor
    record_rank = config.record.record_rank
    assert record_rank != master_rank and record_rank <= world_size - 1
    
    env = make_atari_env(config.solver.env, T.Compose([]), render=False)
    num_actions = env.num_actions()

    ################################################################
    # SEEDING
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    ################################################################
    # AGENT
    if config.solver.dueling:
        network = Dueling_DQN
    else:
        network = Q_Network
    
    q_net = network(input_size=(frames_stack, 84, 84),
                      num_actions=num_actions).to(device)
    
    optimizer_func = cfg_reader.get_optimizer_func(config.server.optimizer)
    
    agent = GorilaDQN_ServerAgent(
        device = device,
        q_net = q_net,
        optimizer_func = optimizer_func,
        shard_factor = shard_factor, 
     )
    
    agent.reset()
    agent.set_optimize_scheme(
        lr=config.server.lr,
        optimize_freq=1,
    )
    agent.set_checkpoint(save_freq, save_path)

    ################################################################
    # SERVICE
    
    server_lock = Lock()
    server = []
    for idx in range(num_servers): 
        server.append(rl_dist.ParamServer(idx, server_lock))
        server[idx].set_listen(4, agent.optimize_counter)
        server[idx].set_param_update(agent.q_net, agent.optimize)
        
    for idx in range(num_servers - 1): server[idx].start()
    print('server current running threads: [%s]' % threading.active_count(), flush=True)
    server[num_servers - 1].run()
    
    
        