import os
import time
from itertools import count
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import rl.utils as utils
from rl.networks import Q_Network
from rl.agents import DQN_Agent


def main():
    ################################################################
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device: [%s]' % device, flush=True)
    
    ################################################################
    # CONFIG
    config = utils.get_config()
    num_episodes = config.solver.episodes
    frame_stack = config.solver.frame_stack
    batch_size = config.replay.sample_batch 
    ################################################################
    # ENVIRONMENT
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1), 
                    T.Resize((84, 84)), 
                    T.ToTensor()])
    env = utils.get_env(config.solver.env, resize,
                        render=config.record.render)
    env.set_frame_stack(num_frames=frame_stack, stack_init_mode='fire')
    env.set_single_life_mode()
    num_actions = env.action_space.n

    ################################################################
    # UTILITIES
    # get threshold between calculated/random action
    get_thres = utils.get_thres_func(eps_start=config.solver.eps_start, 
                                     eps_end=config.solver.eps_end, 
                                     total_num=300)

    ################################################################
    # AGENT  
    q_net = Q_Network(input_size=(frame_stack, 84, 84), 
                      num_actions=num_actions).to(device)
    
    target_net = Q_Network(input_size=(frame_stack, 84, 84), 
                           num_actions=num_actions).to(device)
    
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    loss_func = F.smooth_l1_loss
    optimizer_func = optim.Adam
    lr = config.solver.lr
    # named tuple for experience replay
    memory = namedtuple('SASprimeR', 
                        ('curr_state', 'action', 'next_state', 'reward'))
    replay = utils.get_exp_replay(zipper = memory, 
                                capacity=config.replay.capacity, 
                                sync=config.replay.dist_sync)
    
    agent = DQN_Agent(device = device, 
                      q_net = q_net, 
                      target_net = target_net, 
                      loss_func = loss_func, 
                      optimizer_func = optimizer_func, 
                      lr = lr, 
                      replay = replay)
    agent.reset()
    agent.set_optimize_func(batch_size = batch_size, 
                            gamma = config.solver.gamma)
    
    ################################################################
    # TRAINING 
    for ep in range(num_episodes):
        env.reset()
        thres = get_thres(ep)
        # initial random action
        action = env.sample()
        curr_state, _, done, _ = env.step(action)
        for cycle in count():
            action = agent.next_action(thres, env.sample(), curr_input=curr_state)
            if not done:
                next_state, reward, done, _ = env.step(action)
                exp = agent.replay.zipper(curr_state, action, next_state, reward)
                agent.replay.push(exp)
            else:
                print(time.strftime('[%Y-%m-%d-%H:%M:%S]:'), 
                      'episode [%s/%s], cycle [%s], '
                      'curr threshold [%.5f], total frames [%s]' % (
                          ep + 1, num_episodes, cycle + 1, get_thres(ep), 
                      env.total_step_count), flush=True)
                break
                
            curr_state = next_state
            agent.optimize()
            
        # update the target network, copying all weights and biases in DQN
        if ep % config.solver.update_freq == 0:
            agent.update_target()
            print('[target network updated]', flush=True)
            
        if ep % config.record.freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path, filename='q_net.pth', 
                          obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path, filename='target_net.pth', 
                          obj_name='target_network')
            print('[latest models saved]', flush=True)
            

if __name__ == '__main__':
    main()



