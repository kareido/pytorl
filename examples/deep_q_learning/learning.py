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
    print('current device: [%s]' % DEVICE, flush=True)
    
    ################################################################
    # CONFIG
    config = utils.get_config()
    num_episodes = config.solver.episodes
    frame_step = config.solver.frame_step
    batch_size = config.replay.sample_batch 
    ################################################################
    # ENVIRONMENT
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1), 
                    T.Resize((84, 84)), 
                    T.ToTensor()])
    env = utils.get_env(config.solver.env, resize, device)
    num_actions = env.action_space.n

    ################################################################
    # UTILITIES
    # get threshold between calculated/random action
    get_thres = utils.get_thres_func(eps_start=config.solver.eps_start, 
                                     eps_end=config.solver.eps_end, 
                                     total_num=num_episodes)
    # stack <frame_step> frames to form a neural net input
    get_input = utils.get_stacked_ob_func(env, frame_step)

    ################################################################
    # AGENT  
    q_net = Q_Network(input_size=(frame_step, 84, 84), 
                      num_actions=num_actions).to(device)
    
    target_net = Q_Network(input_size=(frame_step, 84, 84), 
                           num_actions=num_actions).to(device)
    
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=config.solver.lr)
    replay = utils.get_exp_replay(config.replay.capacity, config.replay.dist_sync)
    
    agent = DQN_Agent(device = device, 
                      q_net = q_net, 
                      target_net = target_net, 
                      optimizer = optimizer, 
                      replay = replay)
    agent.reset()
    
    ################################################################
    # TRAINING 
    
    for ep in range(num_episodes):
        env.reset()
        thres = get_thres(ep)
        # initial action
        action = env.sample()
        for cycle in count():
            if config.record.render:
                env.render()
            curr_state = get_input(action)
            action = agent.next_action(thres, env.sample(), curr_input=curr_state)
            _, reward, done, _ = env.step(action.tolist())
            if not done:
                next_state = get_input(action)
                memory = (curr_state, action, next_state, reward)
                agent.replay.push(memory)
            else:
                print(time.strftime('[%y-%m-%d-%H:%M:%S]:'), 
                      'episode [%s/%s], cycle [%s], duration(f) [%s], curr threshold [%.5f]' % (
                        ep + 1, num_episodes, cycle + 1, env.curr_step_count, get_thres(ep)), flush=True)
                break
                
            curr_state = next_state
            
            if len(agent.replay) < batch_size:
#                 print('total_num_frames: [%s]' % (num_frames), flush=True)
                if config.record.debug:
                    print('episode [%s/%s], low memory: [%s/%s]' % (
                        ep + 1, CONFIG.solver.episodes,len(memory), BATCH_SIZE), flush=True)
                continue
                
            sample_mem = agent.replay.sample(batch_size)
            batch_0, batch_1, _, batch_3 = zip(*sample_mem)
#             print(len(batch_0), len(batch_1), len(batch_3), flush=True)
            state_batch = torch.cat(batch_0)
            action_batch = torch.cat(batch_1)
            reward_batch = torch.tensor(batch_3).to(DEVICE)
            state_action_values = agent_net(state_batch).gather(1, action_batch)
#             next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            next_state_values = target_net(state_batch).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * CONFIG.solver.gamma) + reward_batch
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in agent_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if ep % CONFIG.solver.update_freq == 0:
            target_net.load_state_dict(agent_net.state_dict())
            
        if ep % CONFIG.record.freq == 0:
            if not os.path.exists(CONFIG.record.save_path):
                os.makedirs(CONFIG.record.save_path)
            agent_path = os.path.join(CONFIG.record.save_path, 'agent_net.pth')
            target_path = os.path.join(CONFIG.record.save_path, 'target_net.pth')
            torch.save(agent_net.state_dict(), agent_path)
            torch.save(target_net.state_dict(), target_path)
            print('[latest models saved]', flush=True)
#         print('total_num_frames: [%s]' % (num_frames), flush=True)
            

if __name__ == '__main__':
    main()
    print('DONE', flush=True)
