import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from pytorl.agents import GorilaDQN_Agent
import pytorl.distributed as rl_dist
from pytorl.envs import make_atari_env
import pytorl.lib as lib
from pytorl.networks import Dueling_DQN, Q_Network
import pytorl.utils as utils


def ps_proc(master_rank, worker_list):
    ################################################################
    # DEVICE
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('RANK: [%s], current device: [%s]' % (rank, device), flush=True)

    ################################################################
    # CONFIG
    cfg_reader = utils.ConfigReader(default='run_project/config.yaml')
    config = cfg_reader.get_config()
    seed, num_episodes = config.seed, config.solver.episodes

    ################################################################
    # ATARI ENVIRONMENT
    resize = T.Compose(
        [T.ToPILImage(),
        T.Grayscale(1),
        T.Resize((84, 84), interpolation=3),
        T.ToTensor()]
    )
    frames_stack = config.solver.frames_stack
    env = make_atari_env(
        config.solver.env, 
        resize,
        render=config.record.render
    )

    env.set_episodic_init('FIRE')
    env.set_frames_stack(frames_stack)
    env.set_single_life(True)
    env.set_frames_action(config.solver.frames_action)
    num_actions = env.num_actions()
    
    ################################################################
    # UTILITIES
    get_beta = lib.beta_priority_func(
        beta_start=config.replay.beta.start,
        beta_end=config.replay.beta.end,
        num_incres=config.replay.beta.frames,
        global_frames_func=env.global_frames
    )

    get_thres = lib.eps_greedy_func(
        eps_start=config.greedy.start,
        eps_end=config.greedy.end,
        num_decays=config.greedy.frames,
        global_frames_func=env.global_frames
    )
    
    ################################################################
    # SEEDING
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed + rank)
    
    ################################################################
    # AGENT
    if config.solver.dueling:
        network = Dueling_DQN
    else:
        network = Q_Network
    
    q_net = network(input_size=(frames_stack, 84, 84),
                      num_actions=num_actions).to(device)
    
    optimizer_func = cfg_reader.get_optimizer_func(config.solver.optimizer)
    
    agent = GorilaDQN_Agent(
        device = device,
        q_net = q_net,
        optimizer_func = optimizer_func,
     )
    
    agent.reset()
    agent.set_exploration(get_sample=env.sample, get_thres=get_thres)
    agent.set_optimize_scheme(
        lr=config.solver.lr,
        optimize_freq=world_size - 1,
    )

    ################################################################
    # SERVICE
    server = rl_dist.Messenger(rank, master_rank)
    server.set_master_rank(
        agent.shard_idx, 
        agent.padded_len, 
        updates_counter=agent.optimize_counter, 
        worker_list=worker_list
    )
    last_updates = 0
    
    while True:
#         rl_dist.recv_list(param_list)
        server.broadcast_params(agent.q_net.parameters())
#         server.push_params(agent.q_net.parameters())
        sender, shard = server.pull_grad_shard(agent.gradient)
#         print('[master rank %s] receive params from [src rank %s, shard %s]'
#               ', global updates [%s]' % (rank, sender, shard, server.updates_counter()), flush=True)
        agent.optimize()
        if server.updates_counter() % config.record.save_freq == 0 and \
            server.updates_counter() != last_updates:
            agent.save_pth(agent.q_net, config.record.save_path,
                           filename='q_net.pth', obj_name='q_network')
            last_updates = server.updates_counter()

        
        
def test(env, agent):
    env.reset()
    # get initial state
    done = False
    while True:
        action = agent.next_action(env.state)
        next_observ, reward, done, _ = env.step(action)
#         time.sleep(0.02)
        if done: break

    print(time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'],
          'episode [%s/%s], ep-reward [%s], frames [%s]' %
          (env.global_episodes(), num_episodes, env.episodic_reward(),
           env.global_frames()), flush=True)
        
        
        
        
        
        
        
        
        
        
    
    
