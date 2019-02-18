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
    tensorboard = utils.tensorboard_writer(logdir='..')
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
    # AGENT
    q_net = Q_Network(input_size=(frames_stack, 84, 84),
                      num_actions=num_actions)
    utils.init_network(q_net, config.record.load_path, 'q_net.pth', obj_name='q_network')
    q_net = q_net.to(device)
    agent = DQN_Agent(device=device, q_net=q_net)
    agent.set_exploration(env.sample, utils.eps_greedy_func(config.greedy.end, config.greedy.end))

    ################################################################
    # SEEDING
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    ################################################################
    # TESTING
    for _ in range(num_episodes):
        env.reset()
        # get initial state
        done = False
        while True:
            action = agent.next_action(env.state)
            next_observ, reward, done, _ = env.step(action)
            time.sleep(0.02)
            if done: break

        print(time.strftime('[%Y-%m-%d-%H:%M:%S'), '%s]:' % os.environ['run_name'],
              'episode [%s/%s], ep-reward [%s], frames [%s]' %
              (env.global_episodes(), num_episodes, env.episodic_reward(),
               env.global_frames()), flush=True)
        # recording via tensorboard
        tensorboard.add_scalar('episode/reward', env.episodic_reward(), env.global_episodes())


if __name__ == '__main__':
    main()



