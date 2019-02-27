import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import vector_to_parameters
import torchvision.transforms as T
from pytorl.agents import GorilaDQN_ClientAgent
import pytorl.distributed as rl_dist
from pytorl.envs import make_atari_env
import pytorl.lib as lib
from pytorl.networks import Dueling_DQN, Q_Network
import pytorl.utils as utils


def param_client_proc(master_rank, worker_group):
    ################################################################
    # DEVICE
    rank, world_size = dist.get_rank(), dist.get_world_size()
    master_rank = rl_dist.get_master_rank()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('rank: [%s], current device: [%s]' % (rank, device), flush=True)
    
    ################################################################
    # CONFIG
    cfg_reader = utils.ConfigReader(default='run_project/config.yaml')
    config = cfg_reader.get_config()
    seed, num_episodes = config.seed, config.client.episodes
    update_target_freq = config.client.update_target_freq
    gradients_push_freq = config.client.gradients_push_freq
    delay_factor = config.client.delay_factor
    record_rank = config.record.record_rank
    assert record_rank != master_rank and record_rank <= world_size - 1
    
    ################################################################
    # RECORDER
    # tensorboard
    if rank == record_rank:
        print('[rank %s] tensorboard started at specified record rank [%s]' % (rank, rank), flush=True)
        tensorboard = utils.tensorboard_writer(logdir='..')
        tensorboard.add_textfile('config', cfg_reader.config_path)
    else:
        tensorboard = None

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
    env.set_frames_action(config.client.frames_action)
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

    target_net = network(input_size=(frames_stack, 84, 84),
                           num_actions=num_actions).to(device)

    loss_func = cfg_reader.get_loss_func(config.client.loss)

    agent = GorilaDQN_ClientAgent(
        device = device,
        q_net = q_net,
        target_net = target_net,
        loss_func = loss_func,
     )
    
    agent.set_prioritized_replay(
        capacity=config.replay.capacity, 
        batch_size=config.replay.batch_size, 
        init_size=config.replay.init_size, 
        alpha=config.replay.alpha,
        beta_func=get_beta, 
    )
    agent.set_exploration(get_sample=env.sample, get_thres=get_thres)
    agent.set_gradient_scheme(
        gamma=config.client.gamma,
        gradient_freq=1
    )
    agent.set_tensorboard(tensorboard)
    
    ################################################################
    # CLIENT
    client = rl_dist.ParamClient()
    client.set_recv(2)
    client.set_info(agent.shard, agent.gradient_counter)
    client.set_param_update(agent.q_net)
    # q network initialization
    overhead, params = client.recv_param()
    vector_to_parameters(params, agent.q_net.parameters())
    glb_updates, server = overhead
    agent.update_target()

    ################################################################
    # PRETRAIN
    # setting up initial random observations and replays during this session
    print('[rank %s] now about to setup randomized [%s] required initial experience replay...' % (
              rank, agent.replay.init_size), flush=True)
    while True:
        env.reset()
        curr_state, done = env.state().clone(), False
        while len(agent.replay) < agent.replay.init_size and not done:
            action = env.sample()
            next_observ, reward, done, _ = env.step(action)
            next_state = env.state().clone()
            agent.replay.push(curr_state, action, next_state, reward)
            curr_state = next_state

        if not done: break
        # save final action into reply buffer
        agent.replay.push(curr_state, action, None, reward)

    print('[rank %s]' % rank, time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'],
          'prioritized experience replay initialization completed [%s/%s]' % (
          len(agent.replay), agent.replay.init_size), flush=True)

    env.refresh()
    # make a barrier to prevent global timeout problem
    dist.barrier(worker_group)
    
    ################################################################
    # TRAINING
    last_target_update, warned = 0, False
    
    for _ in range(num_episodes):
        env.reset()
        # get initial state
        agent.zero_grad_()
        curr_state, done = env.state().clone(), False
        overhead, params = client.recv_param()
        vector_to_parameters(params, agent.q_net.parameters())
        glb_updates, server = overhead
        while True:
            action = agent.next_action(env.state)
            overhead, params = client.recv_param()
            vector_to_parameters(params, agent.q_net.parameters())
            glb_updates, server = overhead
            if not done:
                next_observ, reward, done, _ = env.step(action)
                next_state = env.state().clone()
            else:
                next_state = None
            agent.replay.push(curr_state, action, next_state, reward)
            # gradient time delay
            glb_avg_time = glb_updates / agent.num_clients
            local_time = agent.gradient_counter() / gradients_push_freq
            if glb_avg_time * (1 - delay_factor) <= local_time:
                warned = False
                agent.backward()
                # push gradient
                if agent.gradient_counter() % gradients_push_freq == 0:
                    client.isend_shard(agent.gradient)
                    agent.zero_grad_()
            else:
                if not warned:
                    warned = True
                    print('[rank %s]' % rank, time.strftime('[%Y-%m-%d-%H:%M:%S'), 
                      '%s]:' % os.environ['run_name'], 'server average '
                      'time [%.1f], local time [%.1f],' % (glb_avg_time, local_time), 
                      'skip gradients due to delay timed out ...', flush=True)
                agent.zero_grad_()
                agent.gradient_counter('add')
            # update target network
            if glb_updates - last_target_update >= update_target_freq: 
                overhead, params = client.recv_param()
                vector_to_parameters(params, agent.target_net.parameters())
                last_target_update = glb_updates
            curr_state = next_state
            if done: break
                
        print('[rank %s]' % rank, time.strftime('[%Y-%m-%d-%H:%M:%S'), 
              '%s]:' % os.environ['run_name'], 'episode [%s/%s], ep-reward [%s], eps [%.2f], '
              'beta [%.2f], timesteps [%s], frames [%s], global_updates [%s], server [%s]' %
              (env.global_episodes(), num_episodes, env.episodic_reward(), get_thres(), get_beta(),
               agent.gradient_counter(), env.global_frames(), glb_updates, server), flush=True)
        
        if tensorboard is not None:
        # recording via tensorboard
            tensorboard.add_scalar('episode/reward', env.episodic_reward(), env.global_episodes())
            tensorboard.add_scalar('episode/thres', get_thres(), env.global_episodes())  

            
            