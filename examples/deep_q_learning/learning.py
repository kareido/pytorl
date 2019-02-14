import os
import random
import time
from itertools import count
import numpy as np
import torch
import torchvision.transforms as T
from rl.agents import DQN_Agent
from rl.envs import gym_env_maker
from rl.networks import Q_Network
import rl.utils as utils


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
    cfg_reader = utils.ConfigReader()
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
    # ENVIRONMENT
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1),
                    T.Resize((84, 84)),
                    T.ToTensor()])
    env = gym_env_maker(config.solver.env, resize,
                        render=config.record.render)
    # seeding
    env.seed(seed)
    env.set_frame_stack(num_frames=frame_stack, stack_init_mode='noop')
    env.set_single_life_mode(True)
    # equivalent to update q-net per skip + 1 frames
    env.set_frameskip_mode(skip=3, discount=config.solver.gamma)
    num_actions = env.action_space.n

    ################################################################
    # AGENT
    q_net = Q_Network(input_size=(frame_stack, 84, 84),
                      num_actions=num_actions).to(device)

    target_net = Q_Network(input_size=(frame_stack, 84, 84),
                           num_actions=num_actions).to(device)

    loss_func = cfg_reader.get_loss_func(config.solver.loss)
    optimizer_func = cfg_reader.get_optimizer_func(config.solver.optimizer)
    lr = config.solver.lr
    replay = utils.NaiveReplay(capacity=config.replay.capacity,
                                obj_format='std_DQN')

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
                            counter=env.get_global_steps)

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
        env.reset(init_mode='fire')
        # initial random action
        action = env.sample()
        curr_state, _, done, _ = env.step(action)
        for cycle in count():
            thres = get_thres(ep, env.total_step_count)
            action = agent.next_action(thres, env.sample(), curr_input=curr_state)
            if not done:
                next_state, reward, done, _ = env.step(action)
                exp = agent.replay.form_obj(curr_state, action.clone(),
                                            next_state, reward.clone())
                agent.replay.push(exp)
                # recording via tensorboard
                tensorboard.add_scalar('step/reward', reward, env.total_step_count)
                tensorboard.add_scalar('step/thres', thres, env.total_step_count)
                tensorboard.add_scalar('replay/size', len(agent.replay),
                                                           env.total_step_count)
            else: break

            curr_state = next_state.clone()
            agent.optimize()
            # potentially update the target network
            framed_update_target(env.total_step_count)

        # potentially update the target network
        episodic_update_target(ep)

        print(time.strftime('[%Y-%m-%d-%H:%M:%S]:'),
              'episode [%s/%s], ep-reward [%s], '
              'eps-thres [%.2f], frames [%s]' % (
              ep + 1, num_episodes, env.episodic_reward.tolist(),
              thres, env.total_step_count), flush=True)
        # recording via tensorboard
        tensorboard.add_scalar('episode/reward', env.episodic_reward.item(), ep + 1)
        tensorboard.add_scalar('episode/thres', thres, ep + 1)

        if ep % config.record.save_freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path,
                           filename='q_net.pth', obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path,
                           filename='target_net.pth', obj_name='target_network')


if __name__ == '__main__':
    main()



