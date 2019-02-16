import os
import random
import time
import numpy as np
import torch
import torchvision.transforms as T
from pytorl.agents import DQN_Agent
from pytorl.envs import make_atari_env
from pytorl.networks import Q_Network
import pytorl.utils as utils

os.environ.setdefault('run_name', 'default')


def main():
    ################################################################
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device: [%s]' % device, flush=True)

    ################################################################
    # CONFIG
    cfg_reader = utils.ConfigReader(default='run_project/atari_config.yaml')
    config = cfg_reader.get_config()
    seed, num_episodes = config.seed, config.solver.episodes

    ################################################################
    # RECORDER
    # tensorboard
    tensorboard = utils.get_tensorboard_writer(logdir='..')
    tensorboard.add_textfile('config', cfg_reader.config_path)

    ################################################################
    # ATARI ENVIRONMENT
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1),
                    T.Resize((84, 84), interpolation=3),
                    T.ToTensor()])
    frames_stack = config.solver.frames_stack
    env = make_atari_env(config.solver.env, resize,
                        render=config.record.render)

    env.set_episodic_init('FIRE')
    env.set_frames_stack(frames_stack)
    env.set_single_life(True)
    env.set_frames_action(4)
    num_actions = env.num_actions()
    
    ################################################################
    # UTILITIES
    replay = utils.VanillaReplay(obj_format='std_DQN', 
                                 capacity=config.replay.capacity,
                                 batch_size=config.replay.batch_size,
                                 init_size=config.replay.init_size)
    
    get_thres = utils.framed_eps_greedy_func(eps_start=config.greedy.start,
                                             eps_end=config.greedy.end,
                                             num_decays=config.greedy.frames,
                                             global_frames_func=env.global_frames)    
    
    ################################################################
    # AGENT
    q_net = Q_Network(input_size=(frames_stack, 84, 84),
                      num_actions=num_actions).to(device)

    target_net = Q_Network(input_size=(frames_stack, 84, 84),
                           num_actions=num_actions).to(device)

    loss_func = cfg_reader.get_loss_func(config.solver.loss)
    optimizer_func = cfg_reader.get_optimizer_func(config.solver.optimizer)

    agent = DQN_Agent(device = device,
                      q_net = q_net,
                      target_net = target_net,
                      loss_func = loss_func,
                      optimizer_func = optimizer_func,
                      replay = replay)
    agent.reset()
    agent.set_exploration(get_sample=env.sample, get_thres=get_thres)
    agent.set_tensorboard(tensorboard)
    agent.set_optimize_scheme(lr=config.solver.lr, 
                              gamma=config.solver.gamma, 
                              optimize_freq=config.solver.optimize_freq, 
                              update_target_freq=config.solver.update_target_freq)
    
    ################################################################
    # SEEDING
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    
    ################################################################
    # PRETRAIN
    # setting up initial random observations and replays during this session
    print('now about to setup randomized [%s] required initial experience replay...' % 
              agent.replay.init_size, flush=True)
    while True:
        env.reset()
        curr_state, done = env.state().clone(), False
        while len(agent.replay) < agent.replay.init_size and not done:
            action = env.sample()
            next_observ, reward, done, _ = env.step(action)
            next_state = env.state().clone()
            exp = agent.replay.form_obj(curr_state, action, next_state, reward)
            agent.replay.push(exp)
            curr_state = next_state
            
        print(time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'], 
              'initializing experience replay progressing [%s/%s]' % (
              len(agent.replay), agent.replay.init_size), flush=True)
        if not done: break
        # save final action into reply buffer
        exp = agent.replay.form_obj(curr_state, action, None, reward)
        agent.replay.push(exp)

    print(time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'], 
          'experience replay initialization completed [%s/%s]' % (
          len(agent.replay), agent.replay.init_size), flush=True)
    
    env.refresh()
    
    ################################################################
    # TRAINING
    for _ in range(num_episodes):
        env.reset()
        # get initial state
        curr_state, done = env.state().clone(), False
        while True:
            action = agent.next_action(env.state)
            if not done:
                next_observ, reward, done, _ = env.step(action)
                next_state = env.state().clone()
            else:
                next_state = None
            exp = agent.replay.form_obj(curr_state, action, next_state, reward)
            agent.replay.push(exp)
            curr_state = next_state
            agent.optimize()
            if done: break

        print(time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'], 
              'episode [%s/%s], ep-reward [%s], eps-thres [%.2f], timesteps [%s], frames [%s]' % 
              (env.global_episodes(), num_episodes, env.episodic_reward(), get_thres(), 
               agent.global_timesteps(), env.global_frames()), flush=True)
        # recording via tensorboard
        tensorboard.add_scalar('episode/reward', env.episodic_reward(), env.global_episodes())
        tensorboard.add_scalar('episode/thres', get_thres(), env.global_episodes())

        if env.global_episodes() % config.record.save_freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path,
                           filename='q_net.pth', obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path,
                           filename='target_net.pth', obj_name='target_network')


if __name__ == '__main__':
    main()



