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


def dqn_proc(master_rank, worker_group):
    ################################################################
    # DEVICE
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('rank: [%s], current device: [%s]' % (rank, device), flush=True)
    
    ################################################################
    # CONFIG
    cfg_reader = utils.ConfigReader(default='run_project/config.yaml')
    config = cfg_reader.get_config()
    seed, num_episodes = config.seed, config.solver.episodes
    update_target_freq = config.solver.update_target_freq

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

    target_net = network(input_size=(frames_stack, 84, 84),
                           num_actions=num_actions).to(device)

    loss_func = cfg_reader.get_loss_func(config.solver.loss)

    agent = GorilaDQN_Agent(
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
    agent.reset()
    agent.set_exploration(get_sample=env.sample, get_thres=get_thres)
    agent.set_gradient_scheme(
        gamma=config.solver.gamma,
        gradient_freq=config.solver.gradient_freq,
    )

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

    ################################################################
    # TRAINING
    client = rl_dist.Messenger(rank, master_rank)
    client.set_learner_rank(agent.shard, config.solver.push_freq)
    last_updates = 0
    dist.barrier()
    for _ in range(num_episodes):
        env.reset()
        # get initial state
        agent.zero_grad_()
        updates = client.pull_broadcast_params(q_net.parameters())
        curr_state, done = env.state().clone(), False
        while True:
#             updates = client.pull_params(q_net.parameters())
#             updates = client.pull_broadcast_params(q_net.parameters())
            action = agent.next_action(env.state)
            if not done:
                next_observ, reward, done, _ = env.step(action)
                next_state = env.state().clone()
            else:
                next_state = None
            agent.replay.push(curr_state, action, next_state, reward)
            agent.backward()
            if client.push_grad_shard(agent.gradient): agent.zero_grad_()
            if updates - last_updates >= update_target_freq: 
                agent.update_target()
                last_updates = updates
            curr_state = next_state
            if done: break
                
        print('[rank %s]' % rank, time.strftime('[%Y-%m-%d-%H:%M:%S'), 
              '%s]:' % os.environ['run_name'], 'episode [%s/%s], ep-reward [%s], eps [%.2f], '
              'beta [%.2f], timesteps [%s], frames [%s], global_updates [%s]' %
              (env.global_episodes(), num_episodes, env.episodic_reward(), get_thres(), get_beta(),
               agent.gradient_counter(), env.global_frames(), updates), flush=True)
        
