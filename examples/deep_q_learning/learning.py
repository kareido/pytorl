import os
import time
from itertools import count
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
    env = utils.get_env(config.solver.env, resize, device, 
                        render=config.record.render)
    env.set_frame_stack(num_frames=frame_stack, stack_init_mode='noop')
    env.set_single_life_mode()
    num_actions = env.action_space.n

    ################################################################
    # UTILITIES
    # get threshold between calculated/random action
    get_thres = utils.get_thres_func(eps_start=config.solver.eps_start, 
                                     eps_end=config.solver.eps_end, 
                                     total_num=num_episodes)

    ################################################################
    # AGENT  
    q_net = Q_Network(input_size=(frame_stack, 84, 84), 
                      num_actions=num_actions).to(device)
    
    target_net = Q_Network(input_size=(frame_stack, 84, 84), 
                           num_actions=num_actions).to(device)
    
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    loss_func = F.smooth_l1_loss
    optimizer = optim.Adam(q_net.parameters(), lr=config.solver.lr)
    replay = utils.get_exp_replay(config.replay.capacity, config.replay.dist_sync)
    
    agent = DQN_Agent(device = device, 
                      q_net = q_net, 
                      target_net = target_net, 
                      loss = loss_func, 
                      optimizer = optimizer, 
                      replay = replay)
    agent.reset()
    
    ################################################################
    # TRAINING 
    for ep in range(num_episodes):
        env.reset()
        thres = get_thres(ep)
        # initial random action
        action = env.sample()
        curr_state, _, done, _ = env.step(action)
#         print('curr_state:', curr_state.shape)
        for cycle in count():
            action = agent.next_action(thres, env.sample(), curr_input=curr_state)
            if not done:
                next_state, reward, done, _ = env.step(action)
#                 print('next_state:', next_state.shape)
                memory = (curr_state, action, next_state, reward)
                agent.replay.push(memory)
            else:
                print(time.strftime('[%y-%m-%d-%H:%M:%S]:'), 
                      'episode [%s/%s], cycle [%s], duration(f) [%s], curr threshold [%.5f]' % (
                        ep + 1, num_episodes, cycle + 1, env.curr_step_count, get_thres(ep)), flush=True)
                break
                
            curr_state = next_state
            
            if len(agent.replay) < batch_size:
                if config.record.debug:
                    print('episode [%s/%s], low memory: [%s/%s]' % (
                        ep + 1, num_episodes,len(agent.replay), batch_size), flush=True)
                continue
                
            sample_mem = agent.replay.sample(batch_size)
            batch_0, batch_1, _, batch_3 = zip(*sample_mem)
            curr_state_batch = torch.cat(batch_0)
#             print(curr_state_batch.shape)
            action_batch = torch.cat(batch_1)
            reward_batch = torch.tensor(batch_3).to(device)
            state_action_values = agent.q_net(curr_state_batch).gather(1, action_batch)
            next_state_values = agent.target_net(curr_state_batch).max(1)[0].detach()
            # compute the expected Q values
            expected_state_action_values = (next_state_values * config.solver.gamma) + reward_batch
            # compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in agent.q_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if ep % config.solver.update_freq == 0:
            agent.update_target()
            
        if ep % config.record.freq == 0:
            agent.save_pth(agent.q_net, config.record.save_path, filename='q_net.pth', 
                          obj_name='q_network')
            agent.save_pth(agent.target_net, config.record.save_path, filename='target_net.pth', 
                          obj_name='target_network')
            print('[latest models saved]', flush=True)
            

if __name__ == '__main__':
    main()
    print('DEEP Q-LEARNING DONE', flush=True)
