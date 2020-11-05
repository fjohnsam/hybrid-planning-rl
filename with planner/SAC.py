import math
import random
import os

import gym
import gym_lunar_lander_modified
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from tqdm import tqdm
from multiprocessing import Process

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = action[0]
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    start_epoch = 0
    rewards = []
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        rewards = checkpoint['rewards']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, rewards

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        

       
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()
        return action[0]


def update1(batch_size,gamma=0.99,soft_tau=1e-2,):
    
    state, action, reward, next_state, done = replay_buffer1.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net11(state, action)
    predicted_q_value2 = soft_q_net21(state, action)
    predicted_value    = value_net1(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net1.evaluate(state)

    target_value = target_value_net1(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion11(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion21(predicted_q_value2, target_q_value.detach())
    soft_q_optimizer11.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer11.step()
    soft_q_optimizer21.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer21.step()
    predicted_new_q_value = torch.min(soft_q_net11(state, new_action),soft_q_net21(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion1(predicted_value, target_value_func.detach())
    value_optimizer1.zero_grad()
    value_loss.backward()
    value_optimizer1.step()
    policy_loss = (log_prob - predicted_new_q_value).mean()
    policy_optimizer1.zero_grad()
    policy_loss.backward()
    policy_optimizer1.step()
    
    for target_param, param in zip(target_value_net1.parameters(), value_net1.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

def update2(batch_size,gamma=0.99,soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer2.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net12(state, action)
    predicted_q_value2 = soft_q_net22(state, action)
    predicted_value    = value_net2(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net2.evaluate(state)


    target_value = target_value_net2(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion12(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion22(predicted_q_value2, target_q_value.detach())
    
    soft_q_optimizer12.zero_grad()
    
    q_value_loss1.backward()
    
    soft_q_optimizer12.step()
    soft_q_optimizer22.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer22.step()    

    predicted_new_q_value = torch.min(soft_q_net12(state, new_action),soft_q_net22(state, new_action))

    target_value_func = predicted_new_q_value - log_prob

    value_loss = value_criterion2(predicted_value, target_value_func.detach())

    value_optimizer2.zero_grad()
    value_loss.backward()
    value_optimizer2.step()

    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer2.zero_grad()
    policy_loss.backward()
    policy_optimizer2.step()

    for target_param, param in zip(target_value_net2.parameters(), value_net2.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

def extract_subplan():
    file = open ('plan.txt','r')
    Lines = file.readlines()
    x=-5
    y=95
    vx=0
    vy=0
    states=[]
    for line in Lines:
        line=line.replace("\n","")
        if line == 'M1S1':
            vx=vx+1
            vy=vy+1
            x=x+vx
            y=y+vy
        elif line == 'M-1S-1':
            vx=vx-1
            vy=vy-1
            x=x+vx
            y=y+vy
        elif line == 'M-1,S1':
            vx=vx-1
            vy=vy+1
            x=x+vx
            y=y+vy
        elif line == 'M1S-1':
            vx=vx+1
            vy=vy-1
            x=x+vx
            y=y+vy
        states.append([x,y,vx,vy])
    subgoal = states[int(len(states)/2)]
    subgoal = [x / 100 for x in subgoal]
    return subgoal;

def first_network_train(subgoal):
    #######################################################################
    env1 = gym.make("LunarLanderContinuous-v2")
    env1 = NormalizedActions(env1)
    action_dim = env1.action_space.shape[0]
    state_dim  = env1.observation_space.shape[0]
    hidden_dim = 256
    ############################ First Network #############################
    value_net1        = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net1 = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net11 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net21 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net1 = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net1.parameters(), value_net1.parameters()):
        target_param.data.copy_(param.data)
        

    value_criterion1  = nn.MSELoss()
    soft_q_criterion11 = nn.MSELoss()
    soft_q_criterion21 = nn.MSELoss()

    value_lr1  = 3e-4
    soft_q_lr1 = 3e-4
    policy_lr1 = 3e-4

    value_optimizer1  = optim.Adam(value_net1.parameters(), lr=value_lr1)
    soft_q_optimizer11 = optim.Adam(soft_q_net11.parameters(), lr=soft_q_lr1)
    soft_q_optimizer21 = optim.Adam(soft_q_net21.parameters(), lr=soft_q_lr1)
    policy_optimizer1 = optim.Adam(policy_net1.parameters(), lr=policy_lr1)

    replay_buffer_size = 1000000
    replay_buffer1 = ReplayBuffer(replay_buffer_size)
    #########################################################################
    ################################## Training first Network ###############
    n_episodes = 5000
    max_frames  = 500000
    max_steps   = 500
    frame_idx  = 0
    rewards     = []
    batch_size  = 128
    start_episode = 0

    prev_difference = None
    curr_difference = None

    policy_net1,policy_optimizer1,frame_idx,rewards = load_checkpoint(policy_net1,policy_optimizer1,'resumeTrain500000fr1')
    print("Resuming from {}".format(frame_idx))
    while frame_idx < max_frames:
        state = env1.reset()
        print(state)
        episode_reward = 0
        
        for step in range(max_steps):
            if frame_idx > 1500:
                action = policy_net1.get_action(state).detach()
                next_state, reward, done, _ = env1.step( action.numpy())
            else:
                action = env1.action_space.sample()
                next_state, reward, done, _ = env1.step(action)

            a = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) 
            b = \
                - 100*np.sqrt((state[0]-subgoal[0])*(state[0]-subgoal[0]) + (state[1]-subgoal[1])*(state[1]-subgoal[1])) \
                - 100*np.sqrt((state[2]-subgoal[2])*(state[2]-subgoal[2]) + (state[3]-subgoal[3])*(state[3]-subgoal[3]))
            curr_difference = a - b 
            if prev_difference is not None:
                reward = reward + prev_difference - curr_difference + 260
            prev_difference = curr_difference

            replay_buffer1.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            # print(len(replay_buffer1))

            if len(replay_buffer1) > batch_size:
                # print(len(replay_buffer1))
                # print(replay_buffer1)
                # update1(batch_size)
                gamma=0.99
                soft_tau=1e-2
                state_, action_, reward_, next_state_, done_ = replay_buffer1.sample(batch_size)

                state_      = torch.FloatTensor(state_).to(device)
                next_state_ = torch.FloatTensor(next_state_).to(device)
                action_     = torch.FloatTensor(action_).to(device)
                reward_     = torch.FloatTensor(reward_).unsqueeze(1).to(device)
                done_       = torch.FloatTensor(np.float32(done_)).unsqueeze(1).to(device)

                predicted_q_value1_ = soft_q_net11(state_, action_)
                predicted_q_value2_ = soft_q_net21(state_, action_)
                predicted_value_    = value_net1(state_)
                new_action_, log_prob_, epsilon_, mean_, log_std_ = policy_net1.evaluate(state_)

                target_value_ = target_value_net1(next_state_)
                target_q_value_ = reward_ + (1 - done_) * gamma * target_value_
                # print("check1")
                q_value_loss1_ = soft_q_criterion11(predicted_q_value1_, target_q_value_.detach())
                q_value_loss2_ = soft_q_criterion21(predicted_q_value2_, target_q_value_.detach())
                soft_q_optimizer11.zero_grad()
                q_value_loss1_.backward()
                soft_q_optimizer11.step()
                soft_q_optimizer21.zero_grad()
                q_value_loss2_.backward()
                soft_q_optimizer21.step()
                predicted_new_q_value_ = torch.min(soft_q_net11(state_, new_action_),soft_q_net21(state_, new_action_))
                target_value_func_ = predicted_new_q_value_ - log_prob_
                # print("check2")
                value_loss_ = value_criterion1(predicted_value_, target_value_func_.detach())
                # print("check2")
                value_optimizer1.zero_grad()
                value_loss_.backward()
                value_optimizer1.step()
                policy_loss_ = (log_prob_ - predicted_new_q_value_).mean()
                policy_optimizer1.zero_grad()
                policy_loss_.backward()
                policy_optimizer1.step()
                
                for target_param, param in zip(target_value_net1.parameters(), value_net1.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )

            if done:
                break
        state = {'epoch': frame_idx + 1, 'state_dict': policy_net1.state_dict(),
                    'optimizer': policy_optimizer1.state_dict(), 'rewards': rewards}
        torch.save(state, 'resumeTrain500000fr1')
        start_episode+=1

        print(" \r first network : frame {} reward: {}".format(frame_idx,episode_reward))
        rewards.append(episode_reward)
    plot(frame_idx, rewards)
    torch.save(policy_net1,'Train500000fr1')

def second_network_train(subgoal):
    env2 = gym.make("LunarLanderContinuous-v2")
    env2 = NormalizedActions(env2)
    action_dim = env2.action_space.shape[0]
    state_dim  = env2.observation_space.shape[0]
    hidden_dim = 256
    ############################### Second Network ##########################
    value_net2        = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net2 = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net12 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net22 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net2 = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net2.parameters(), value_net2.parameters()):
        target_param.data.copy_(param.data)

    value_criterion2  = nn.MSELoss()
    soft_q_criterion12 = nn.MSELoss()
    soft_q_criterion22 = nn.MSELoss()

    value_lr2  = 3e-4
    soft_q_lr2 = 3e-4
    policy_lr2 = 3e-4

    value_optimizer2  = optim.Adam(value_net2.parameters(), lr=value_lr2)
    soft_q_optimizer12 = optim.Adam(soft_q_net12.parameters(), lr=soft_q_lr2)
    soft_q_optimizer22 = optim.Adam(soft_q_net22.parameters(), lr=soft_q_lr2)
    policy_optimizer2 = optim.Adam(policy_net2.parameters(), lr=policy_lr2)


    replay_buffer_size = 1000000
    replay_buffer2 = ReplayBuffer(replay_buffer_size)
    
    ###################################### Training Second Network ################
    n_episodes = 5000
    max_frames  = 500000
    max_steps   = 500
    frame_idx  = 0
    rewards     = []
    batch_size  = 128
    start_episode = 0
    prev_difference=None

    policy_net2,policy_optimizer2,frame_idx,rewards = load_checkpoint(policy_net2,policy_optimizer2,'resumeTrain500000fr2')
    print("Resuming from {}".format(frame_idx))
    while frame_idx < max_frames:
        state = env2.reset()
        print(state)
        episode_reward = 0
        for step in range(max_steps):
            if frame_idx > (1500+(500000/2)):
                action = policy_net2.get_action(state).detach()
                next_state, reward, done, _ = env2.step(action.numpy())
            else:
                action = env2.action_space.sample()
                next_state, reward, done, _ = env2.step(action)
            # print(subgoal)
            a = \
                - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
                - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) 
            b = \
                - 100*np.sqrt((state[0]-subgoal[0])*(state[0]-subgoal[0]) + (state[1]-subgoal[1])*(state[1]-subgoal[1])) \
                - 100*np.sqrt((state[2]-subgoal[2])*(state[2]-subgoal[2]) + (state[3]-subgoal[3])*(state[3]-subgoal[3]))
            curr_difference = a - b 
            # print("a=",a)
            # print("b=",b)
            # print("previous diff=",prev_difference)
            # print("current diff=", curr_difference)
            if prev_difference is not None:
                reward = reward + prev_difference - curr_difference
            prev_difference = curr_difference
            # print("reward=",reward)
            replay_buffer2.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            if len(replay_buffer2) > batch_size:
                # update2(batch_size)
                gamma=0.99
                soft_tau=1e-2
                state_, action_, reward_, next_state_, done_ = replay_buffer2.sample(batch_size)

                state_      = torch.FloatTensor(state_).to(device)
                next_state_ = torch.FloatTensor(next_state_).to(device)
                action_     = torch.FloatTensor(action_).to(device)
                reward_     = torch.FloatTensor(reward_).unsqueeze(1).to(device)
                done_       = torch.FloatTensor(np.float32(done_)).unsqueeze(1).to(device)

                predicted_q_value1_ = soft_q_net12(state_, action_)
                predicted_q_value2_ = soft_q_net22(state_, action_)
                predicted_value_    = value_net2(state_)
                new_action_, log_prob_, epsilon_, mean_, log_std_ = policy_net2.evaluate(state_)


                target_value_ = target_value_net2(next_state_)
                target_q_value_ = reward_ + (1 - done_) * gamma * target_value_
                q_value_loss1_ = soft_q_criterion12(predicted_q_value1_, target_q_value_.detach())
                q_value_loss2_ = soft_q_criterion22(predicted_q_value2_, target_q_value_.detach())
                
                soft_q_optimizer12.zero_grad()
                
                q_value_loss1_.backward()
                
                soft_q_optimizer12.step()
                soft_q_optimizer22.zero_grad()
                q_value_loss2_.backward()
                soft_q_optimizer22.step()    

                predicted_new_q_value_ = torch.min(soft_q_net12(state_, new_action_),soft_q_net22(state_, new_action_))

                target_value_func_ = predicted_new_q_value_ - log_prob_

                value_loss_ = value_criterion2(predicted_value_, target_value_func_.detach())

                value_optimizer2.zero_grad()
                value_loss_.backward()
                value_optimizer2.step()

                policy_loss_ = (log_prob_ - predicted_new_q_value_).mean()

                policy_optimizer2.zero_grad()
                policy_loss_.backward()
                policy_optimizer2.step()

                for target_param, param in zip(target_value_net2.parameters(), value_net2.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )
            # print(done)
            if done:
                break
        state = {'epoch': frame_idx + 1, 'state_dict': policy_net2.state_dict(),
                    'optimizer': policy_optimizer2.state_dict(), 'rewards': rewards}
        torch.save(state, 'resumeTrain500000fr2')
        start_episode+=1

        print("\r second network : frame {} reward: {}".format(frame_idx,episode_reward))
        rewards.append(episode_reward)
    plot(frame_idx, rewards)
    torch.save(policy_net2,'Train500000fr2')


def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(start_episode):
        patch.set_data(frames[start_episode])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(anim)

def test():
    # env = gym.make("LunarLanderContinuous-v2")
    env = NormalizedActions(gym.make("lunar_lander_modified-v0"))

    # Run a demo of the environment
    state = env.reset()
    # print(state)
    cum_reward = 0
    frames = []
    for t in range(50000):
        # Render into buffer. 
        # env.render(mode = 'rgb_array')
        frames.append(env.render(mode = 'rgb_array'))
        action = policy_net.get_action(state)
        state, reward, done, info = env.step(action.detach().numpy())
        if done:
            break
    env.close()
    display_frames_as_gif(frames)

if __name__ == '__main__':
    ###############################################################################
    subgoal = extract_subplan()
    P1 = Process(target=first_network_train,args=(subgoal,))
    # subgoal=[0.0,0.0,0.0,0.0]
    P2 = Process(target=second_network_train,args=(subgoal,))
    print("Strting process1")
    # P1.start()
    print("Strting process2")
    P2.start()
    # P1.join()
    P2.join()