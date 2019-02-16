import os
import random
import time
from itertools import count
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from rl.agents import DQN_Agent
from rl.envs import make_atari_env, make_ctrl_env
from rl.networks import Q_Network, Q_MLP
import rl.utils as utils


os.environ.setdefault('run_name', 'default')


"""
controller for when to update target
"""
def update_target_controller(agent, freq, mode, cfg_mode, debug=False):
    assert mode in {'episodic', 'framed'}
    assert cfg_mode in {'episodic', 'framed'}
    def _controller(num):
        if mode == cfg_mode and num % freq == 0:
            agent.update_target()
            if debug:
                print('target network updated at %s [%s]' % (
                        mode, num), flush=True)
    return _controller


def thres_controller(start, end, steps, delay, decay, mode, num_episodes):
    assert mode in {'episodic', 'framed'}
    if not steps:
        if mode == 'framed':
            raise ValueError('steps must be specified under framed mode')
        steps = num_episodes
    func = utils.get_epsilon_greedy_func(eps_start=start, eps_end=end,
                                     delay=delay, steps=steps, decay=decay)
    if mode == 'episodic':
        def _controller(ep, frame):
            return func(ep)
    else:
        def _controller(ep, frame):
            return func(frame)
    return _controller


def main():
    ################################################################
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device: [%s]' % device, flush=True)

    ################################################################
    # CONFIG
    cfg_reader = utils.ConfigReader(default='run_proj/atari_config.yaml')
    config = cfg_reader.get_config()
    seed = config.seed
    num_episodes = config.solver.episodes
    frame_stack = config.solver.frame_stack
    batch_size = config.replay.sample_batch

    # seeding
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    ################################################################
    # RECORDER
    # tensorboard
    tensorboard = utils.get_tensorboard_writer(logdir='..')
    tensorboard.add_textfile('config', cfg_reader.config_path)

    ################################################################
#     ATARI ENVIRONMENT
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1),
                    T.Resize((84, 84), interpolation=Image.CUBIC),
                    T.ToTensor()])
    env = make_atari_env(config.solver.env, resize,
                        render=config.record.render)
    # seeding
    env.seed(seed)
    env.set_episodic_init('FIRE')
    env.set_frames_stack(frame_stack)
    env.set_single_life(True)
    env.set_frames_action(4)
    num_actions = env.num_actions()
    
    ################################################################
    # CLASSIC CONTROL ENVIRONMENT
#     env = make_ctrl_env('CartPole-v1', render=config.record.render)
#     # seeding
#     env.seed(seed)
#     env.set_frames_stack(frame_stack)
#     env.set_frames_action(1)
#     num_actions = env.num_actions()

    ################################################################
    # AGENT
    q_net = Q_Network(input_size=(frame_stack, 84, 84),
                      num_actions=num_actions).to(device)

    target_net = Q_Network(input_size=(frame_stack, 84, 84),
                           num_actions=num_actions).to(device)

#     q_net = Q_MLP(input_size=(frame_stack, env.observ_shape()),
#                       num_actions=num_actions).to(device)

#     target_net = Q_MLP(input_size=(frame_stack, env.observ_shape()),
#                       num_actions=num_actions).to(device)

    loss_func = cfg_reader.get_loss_func(config.solver.loss)
    optimizer_func = cfg_reader.get_optimizer_func(config.solver.optimizer)
    lr = config.solver.lr
    replay = utils.NaiveReplay(obj_format='std_DQN', capacity=config.replay.capacity)

    agent = DQN_Agent(device = device,
                      q_net = q_net,
                      target_net = target_net,
                      loss_func = loss_func,
                      optimizer_func = optimizer_func,
                      lr = lr,
                      replay = replay)
    agent.reset()
    agent.set_optimize_func(batch_size=config.replay.sample_batch,
                            gamma=config.solver.gamma,
                            min_replay=config.replay.init_num,
                            learn_freq=config.solver.learn_freq,
                            tensorboard=tensorboard,
                            counter=env.global_frames)

    ################################################################
    # FUNCTION SUPPORTS
    # get threshold between calculated/random action
    get_thres = thres_controller(
                        config.greedy.start,
                        config.greedy.end,
                        config.greedy.steps,
                        config.greedy.delay,
                        config.greedy.decay,
                        config.greedy.mode,
                        num_episodes)
    # get target update controller
    episodic_update_target = update_target_controller(
                                agent,
                                config.solver.target.update_freq,
                                'episodic',
                                config.solver.target.update_mode,
                                debug=config.record.debug
                                )
    framed_update_target = update_target_controller(
                                agent,
                                config.solver.target.update_freq,
                                'framed',
                                config.solver.target.update_mode,
                                debug=config.record.debug
                                )

    ################################################################
    # TRAINING
    for ep in range(num_episodes):
        env.reset()
        # initial random action
        action = env.sample()
        curr_observ, _, done, _ = env.step(action)
        curr_state = env.state().clone()
        done = False
        while not done:
            thres = get_thres(ep, env.global_frames())
            action = agent.next_action(thres, env.sample(), curr_input=env.state())
            if not done:
                next_observ, reward, done, _ = env.step(action)
                next_state = env.state().clone()
            else:
                next_state = None
                
            exp = agent.replay.form_obj(curr_state, action, next_state, reward)
            agent.replay.push(exp)
            
                # recording via tensorboard
#                 tensorboard.add_scalar('step/reward', env.action_reward(), env.global_frames())
            tensorboard.add_scalar('step/thres', thres, env.global_frames())
            tensorboard.add_scalar('replay/size', len(agent.replay),
                                                       env.global_frames())

            curr_state = next_state
            agent.optimize()
            # potentially update the target network
            framed_update_target(env.global_frames())

        # potentially update the target network
        episodic_update_target(ep)

        print(time.strftime('[%Y-%m-%d-%H:%M:%S]'), 
              '[%s]:' % os.environ['run_name'], 
              'episode [%s/%s], ep-reward [%s], '
              'eps-thres [%.2f], frames [%s]' % (
              ep + 1, num_episodes, env.episodic_reward(),
              thres, env.global_frames()), flush=True)
        # recording via tensorboard
        tensorboard.add_scalar('episode/reward', env.episodic_reward(), ep + 1)
        tensorboard.add_scalar('episode/thres', thres, ep + 1)

        if ep % config.record.save_freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path,
                           filename='q_net.pth', obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path,
                           filename='target_net.pth', obj_name='target_network')


if __name__ == '__main__':
    main()



