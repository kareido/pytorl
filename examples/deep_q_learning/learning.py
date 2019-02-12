import os
import time
from itertools import count
import torch
import torchvision.transforms as T
import rl.utils as utils
from rl.networks import Q_Network
from rl.agents import DQN_Agent


"""
controller for when to update target
"""
def update_target_controller(agent, freq, mode, cfg_mode):
    assert mode in {'episodic', 'framed'}
    assert cfg_mode in {'episodic', 'framed'}
    def _controller(num):
        if mode == cfg_mode and num % freq == 0:
            agent.update_target()
            print('target network updated at %s [%s]' % (mode, num), flush=True)
    return _controller


def thres_controller(start, end, steps, mode, num_episodes):
    assert mode in {'episodic', 'framed'}
    if not steps: 
        if mode == 'framed':
            raise ValueError('steps must be specified under framed mode')
        steps = num_episodes
    func = utils.get_thres_func(eps_start=start, eps_end=end, steps=steps)
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
    # AGENT  
    q_net = Q_Network(input_size=(frame_stack, 84, 84), 
                      num_actions=num_actions).to(device)
    
    target_net = Q_Network(input_size=(frame_stack, 84, 84), 
                           num_actions=num_actions).to(device)
    
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
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
    agent.set_optimize_func(batch_size = batch_size, 
                            gamma = config.solver.gamma)
    
    ################################################################
    # UTILITIES
    # get threshold between calculated/random action
    get_thres = thres_controller(
                        config.greedy.start, 
                        config.greedy.end, 
                        config.greedy.steps, 
                        config.greedy.mode, 
                        num_episodes)
    # get target update controller
    episodic_update_target = update_target_controller(
                                agent, 
                                config.solver.target.update_freq, 
                                'episodic', 
                                config.solver.target.update_mode
                                )
    framed_update_target = update_target_controller(
                                agent, 
                                config.solver.target.update_freq, 
                                'framed', 
                                config.solver.target.update_mode
                                )

    ################################################################
    # TRAINING 
    for ep in range(num_episodes):
        env.reset()
        thres = get_thres(ep, env.total_step_count)
        # initial random action
        action = env.sample()
        curr_state, _, done, _ = env.step(action)
        for cycle in count():
            action = agent.next_action(thres, env.sample(), curr_input=curr_state)
            if not done:
                next_state, reward, done, _ = env.step(action)
                exp = agent.replay.form_obj(curr_state, action, next_state, reward)
                agent.replay.push(exp)
            else:
                print(time.strftime('[%Y-%m-%d-%H:%M:%S]:'), 
                      'episode [%s/%s], cycle [%s], '
                      'curr threshold [%.5f], total frames [%s]' % (
                          ep + 1, num_episodes, cycle + 1, 
                          get_thres(ep, env.total_step_count), 
                          env.total_step_count), flush=True)
                break
                
            curr_state = next_state
            agent.optimize()
            # potentially update the target network
            framed_update_target(env.total_step_count)
            
        # potentially update the target network
        episodic_update_target(ep)
            
        if ep % config.record.save_freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path, filename='q_net.pth', 
                          obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path, filename='target_net.pth', 
                          obj_name='target_network')
            print('[latest models saved]', flush=True)
            

if __name__ == '__main__':
    main()



