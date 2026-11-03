from env import Env
from render import Render
import torch.nn as nn
from base_agent import BaseAgent
import torch
import numpy as np
import random
from springc_utils import toTensor, toNumpy
import torch.optim as optim
from tqdm import tqdm
from repalybuffer import ReplayBuffer
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

random.seed(111143)


class PolicyNet(nn.Module):

    def __init__(self, input_dim=2, b_action=4):
        super().__init__()
        mid_dim = input_dim*10
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, b_action)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x



class Agent(BaseAgent):



    def __init__(self, rows, cols, n_actions, gamma, device, batch_size=32, max_buffer_num=10000):
        super().__init__(rows, cols, n_actions, gamma)
        input_dim = rows * cols
        self.policy_net = PolicyNet(input_dim, n_actions).to(device)
        self.policy_net.train()
        self.device = device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.n_episode = 1000

    
    def select_action(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            prob = self.policy_net(state)
            action = Categorical(prob).sample()
        self.policy_net.train()
        return action.item()
            
        

    # def get_policy(self, state_vector):
    #     states = [[r, c] for r in range(self.rows) for c in range(self.cols)]
    #     states = toTensor(self.env.st).to(self.device).float()
    #     policy = self.target_net(states).argmax(dim=-1).reshape(-1, self.cols)
    #     return policy
    
    
    
    def train_one_episode(self, trans):
        states = trans['states']
        actions = trans['actions']
        rewards = trans['rewards']
        G = 0
        self.optimizer.zero_grad()
        episode_return = 0
        for i in reversed(range(len(states))):
            state, action, reward = states[i], actions[i], rewards[i]
            G = reward + G * self.gamma
            loss = -1 * G * torch.log(self.policy_net(state)[0, action])
            loss.backward()
            episode_return += reward
        self.optimizer.step()
        return episode_return
            
        


def train():
    device = torch.device("cuda")
    env = Env()
    render = Render(env)
    rows, cols, n_actions = env.rows, env.cols, env.n_actions
    gamma = 0.95
    agent = Agent(rows, cols, n_actions, gamma, device)
    n_episode = 4000
    env.reset()
    writter = W
    state_vector = toTensor(env.state_vector).float().to(device)
    for i in tqdm(range(n_episode)):
        done = False
        
        state = env.pos2state(env.cur_pos).reshape(1, -1)
        state = toTensor(state, device).float()
        trans = {'states':[], 'actions':[], 'rewards':[], 'next_states':[], 'dones':[]}
        while not done:
            action = agent.select_action(state)
            next_state, r, done = env.step(action)
            next_state = toTensor(next_state.reshape(1, -1), device).float()
            trans['states'].append(state)
            trans['actions'].append(action)
            trans['rewards'].append(r)
            trans['next_states'].append(next_state)
            trans['dones'].append(done)
            state = next_state
        episode_return = agent.train_one_episode(trans)
            

train()

        
        


    