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
from datetime import datetime

log_dir = f"runs/A2C_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class PolicyNet(nn.Module):

    def __init__(self, input_dim=2, mid_dim=128, n_action=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, n_action)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


class ValueNet(nn.Module):

    def __init__(self, input_dim=2, mid_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, 1)
        self.act = nn.SiLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class Agent(BaseAgent):



    def __init__(self, rows, cols, n_actions, gamma, device, batch_size=32, max_buffer_num=10000):
        super().__init__(rows, cols, n_actions, gamma)
        input_dim = rows * cols
        self.policy_net = PolicyNet(input_dim, n_action=n_actions).to(device)
        self.policy_net.train()
        self.value_net = ValueNet(input_dim).to(device)
        self.value_net.train()
        self.device = device
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=1e-2)
        self.n_episode = 1000
        self.mse_loss = nn.MSELoss()

    
    def select_action(self, state):
        with torch.no_grad():
            prob = self.policy_net(state)
            action = Categorical(prob).sample()
        return action.item()
            
        

    def get_policy(self, states):
        states = toTensor(states).to(self.device).float()
        policy = self.policy_net(states).argmax(dim=-1).reshape(-1, self.cols)
        return policy
    
    
    
    def train_one_episode(self, trans):
        states = trans['states']
        actions = trans['actions']
        rewards = trans['rewards']
        next_states = trans['next_states']
        dones = trans['dones']
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        actions = toTensor(actions).to(self.device).reshape(-1, 1)
        dones = toTensor(dones).to(self.device).reshape(-1, 1).float()
        rewards = toTensor(rewards).to(self.device).reshape(-1, 1)
        td_target = self.value_net(next_states) * (1-dones) * self.gamma + rewards
        td_predict = self.value_net(states)
        delta = td_target - td_predict
        logprob = torch.log(self.policy_net(states).gather(1, actions))

        critic_loss = self.mse_loss(td_predict, td_target.detach())
        actor_loss = torch.mean(-logprob * delta.detach())
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_value.step()


def train():
    device = torch.device("cuda")
    env = Env()
    render = Render(env)
    rows, cols, n_actions = env.rows, env.cols, env.n_actions
    gamma = 0.95
    agent = Agent(rows, cols, n_actions, gamma, device)
    n_episode = rows * cols * 100
    writter = SummaryWriter(log_dir=log_dir)
    state_vector = toTensor(env.state_vector).float().to(device)
    
    for i in tqdm(range(100)):
        for sr in range(rows):
            for sc in range(cols):
                env.reset()
                # env.cur_pos = (sr, sc)
                done = False
                state = env.pos2state(env.cur_pos).reshape(1, -1)
                state = toTensor(state, device).float()
                trans = {'states':[], 'actions':[], 'rewards':[], 'next_states':[], 'dones':[]}
                episode_return = 0
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
                    episode_return += r
                agent.train_one_episode(trans)
                writter.add_scalar("Reward/episode_return", episode_return, i+1)
                writter.add_scalar("Steps/episode_steps", env.n_step, i+1)
        if i%10==0:
            policy = agent.get_policy(state_vector)
            render.render_policy(policy)
    print(f"最终的策略：\n")
    policy = agent.get_policy(state_vector)
    render.render_policy(policy)
        
            

train()

        
        


    