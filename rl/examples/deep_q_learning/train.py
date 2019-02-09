import os
import time
import random
import math
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import rl
import rl.utils as util
import rl.networks as network


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBSIZE = T.Compose([T.ToPILImage(),
                T.Grayscale(1), 
                T.Resize((84, 84)), 
                T.ToTensor()])

CONFIG = util.get_config()
BATCH_SIZE = CONFIG.replay.sample_batch
FRAME_STEP = CONFIG.solver.frame_step


def curr_thres(config, num_frames):
    return config.solver.eps_end + (config.solver.eps_start - config.solver.eps_end) * \
            math.exp(-1. * num_frames / config.solver.eps_decay)


def select_action(config, agent_net, curr_state, num_frames, num_actions):
    sample = random.random()
    eps_threshold = curr_thres(config, num_frames)
    if sample > eps_threshold and curr_state is not None:
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        with torch.no_grad():
            curr_q = agent_net(curr_state)
            return curr_q.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_actions)]], device=DEVICE, dtype=torch.long)

    
def k_frames(env, action ,k):
    ret = None
    for _ in range(k):
        ob, _, done, _ = env.step(action)
        if ret is None:
            ret = ob.clone()
        else:
            ret = torch.cat((ret, ob))
#             print(ret.shape)
        if done:
            while ret.shape[0] < k:
                ret = torch.cat((ret, ob))
            break
    
    return ret.unsqueeze(0)
        
    
def main():
    env = util.get_env(CONFIG.solver.env, OBSIZE, DEVICE)
    num_actions = env.action_space.n
    agent_net = network.DQN.Q_Network(input_size=(FRAME_STEP, 84, 84), num_actions=num_actions).to(DEVICE)
    target_net = network.DQN.Q_Network(input_size=(FRAME_STEP, 84, 84), num_actions=num_actions).to(DEVICE)
    target_net.load_state_dict(agent_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(agent_net.parameters(), lr=CONFIG.solver.lr)
    memory = util.get_exp_replay(CONFIG.replay.capacity, CONFIG.replay.dist_sync)
    
    num_frames = 0
    for ep in range(CONFIG.solver.episodes):
        ob = env.reset()
        action = select_action(CONFIG, agent_net, None, num_frames, num_actions)
        num_frames += FRAME_STEP
        for f in count():
            env.render()
            curr_state = k_frames(env, action ,FRAME_STEP)
            action = select_action(CONFIG, agent_net, curr_state, num_frames, num_actions)
            _, reward, done, _ = env.step(action.item())
            if not done:
                num_frames += FRAME_STEP
                next_state = k_frames(env, action ,FRAME_STEP)
                transition = (curr_state, action, next_state, reward)
                memory.push(transition)
            else:
                print(time.strftime('[%m-%d-%H:%M:%S]: '
                       ) + 'episode [%s/%s], duration [%s], total frames [%s], curr threshold [%.5f]' % (
                        ep + 1, CONFIG.solver.episodes, f + 1, num_frames, curr_thres(CONFIG, num_frames)), flush=True)
                break
                
            curr_state = next_state

            if len(memory) < BATCH_SIZE:
#                 print('total_num_frames: [%s]' % (num_frames), flush=True)
                print('episode [%s/%s], low memory: [%s/%s]' % (
                        ep + 1, CONFIG.solver.episodes,len(memory), BATCH_SIZE), flush=True)
                continue
#             if num_frames < 10000:
#                 continue
            sample_trans = memory.sample(BATCH_SIZE)
            batch_0, batch_1, _, batch_3 = zip(*sample_trans)
#             print(len(batch_0), len(batch_1), len(batch_3), flush=True)
            state_batch = torch.cat(batch_0)
            action_batch = torch.cat(batch_1)
            reward_batch = torch.tensor(batch_3)
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
            agent_path = os.path.join(CONFIG.record.save_path, 'agent_net.pth')
            target_path = os.path.join(CONFIG.record.save_path, 'target_net.pth')
            torch.save(agent_net.state_dict(), agent_path)
            torch.save(target_net.state_dict(), target_path)
            print('[latest models saved]', flush=True)
#         print('total_num_frames: [%s]' % (num_frames), flush=True)
            

if __name__ == '__main__':
    main()
    print('DONE')