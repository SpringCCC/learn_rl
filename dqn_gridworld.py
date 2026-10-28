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

class SimpleNet(nn.Module):

    def __init__(self, input_dim=2, out_dim=5):
        super().__init__()
        mid_dim = input_dim*10
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class Agent(BaseAgent):



    def __init__(self, rows, cols, n_actions, gamma, device):
        super().__init__(rows, cols, n_actions, gamma)
        self.target_net = SimpleNet(2, n_actions).to(device)
        self.main_net = SimpleNet(2, n_actions).to(device)
        self.syncNet()
        self.target_net.eval()
        self.reply_buffer = []
        self.device = device



    def select_action(self):
        return np.random.randint(self.n_actions)


    def syncNet(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def get_policy(self):
        states = [[r, c] for r in range(self.rows) for c in range(self.cols)]
        states = toTensor(states).to(self.device).float()
        policy = self.target_net(states).argmax(dim=-1).reshape(-1, self.cols)
        return policy



def parse_raw_data(raw_train_data, device=None):
    state = []
    next_state = []
    rewards = []
    actions = []
    for data in raw_train_data:
        s0,a0,r,s1 = data
        state.append(s0)
        actions.append(a0)
        rewards.append(r)
        next_state.append(s1)
    state = toTensor(state, device).float()
    next_state = toTensor(next_state, device).float()
    rewards = toTensor(rewards, device).float()
    actions = toTensor(actions, device).reshape(-1, 1).long()
    return state, actions, rewards, next_state



def train():
    device = torch.device("cuda")
    env = Env()
    render = Render(env)
    rows, cols, n_actions = env.rows, env.cols, env.n_actions
    gamma = 0.9
    agent = Agent(rows, cols, n_actions, gamma, device)
    episode_length = 10000
    n_update = 100
    n_train_steps = 10000
    batch_num = 32
    ## 构建replay
    env.reset()
    visit = np.zeros((rows, cols), dtype=np.int32)
    for i in range(episode_length):
        s0 = env.cur_pos
        a0 = agent.select_action()
        s1, reward, done = env.step(a0)
        agent.reply_buffer.append((s0, a0, reward, s1))
        visit[s1]+=1
    render.render_value(visit)

    loss_fuc = nn.MSELoss()
    optimizer = optim.Adam(agent.main_net.parameters(), lr=0.001)
    pbar = tqdm(total=n_train_steps)
    visit *=0
    ## train
    for i in range(n_train_steps):
        raw_train_data = random.sample(agent.reply_buffer, batch_num)
        state, actions, rewards, next_state = parse_raw_data(raw_train_data, device)
        ###
        ns = toNumpy(state).astype(np.int32)
        ns = [tuple(a) for a in ns]
        for a in ns:
            visit[a] += 1
        ####
        with torch.no_grad():
            td_target = agent.target_net(next_state).max(dim=1)[0] * agent.gamma + rewards
        pred = agent.main_net(state).gather(1, actions).squeeze()
        loss = loss_fuc(pred, td_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)% n_update==0:
            agent.syncNet()
        pbar.update(1)
        pbar.set_description(f"Steps:{i+1}/{n_train_steps}")
    render.render_value(visit)
    agent.syncNet()
    print("训练完成")
    policy = toNumpy(agent.get_policy())
    render.render_policy(policy)
    

train()

        
        


    