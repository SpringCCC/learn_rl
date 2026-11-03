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
random.seed(111143)
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



    def __init__(self, rows, cols, n_actions, gamma, device, batch_size=32, max_buffer_num=10000):
        super().__init__(rows, cols, n_actions, gamma)
        input_dim = rows * cols
        self.target_net = SimpleNet(input_dim, n_actions).to(device)
        self.main_net = SimpleNet(input_dim, n_actions).to(device)
        self.syncNet()
        self.target_net.eval()
        self.main_net.train()
        self.reply_buffer = []
        self.device = device
        self.epsilon_min=0.05
        self.epsilon_max=0.95
        self.epsilon_decay=0.995
        self.epsilon = self.epsilon_max
        self.memory = ReplayBuffer(max_buffer_num, batch_size, device)
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-3)
        self.train_step_count = 0
        self.syncnet_freq = 10

    # def select_action(self, state):
    #     if np.random.random() < self.epsilon:
    #         return np.random.randint(self.n_actions)
    #     self.main_net.eval()
    #     state = toTensor(state)[None].to(self.device)
    #     with torch.no_grad():
    #         action = self.main_net(state).argmax(dim=-1).item()
    #     self.main_net.train()
    #     return action
    
    
    def select_action(self, state):
        return np.random.randint(self.n_actions)


    def syncNet(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def get_policy(self, state_vector):
        states = [[r, c] for r in range(self.rows) for c in range(self.cols)]
        states = toTensor(self.env.st).to(self.device).float()
        policy = self.target_net(states).argmax(dim=-1).reshape(-1, self.cols)
        return policy
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample()
        q_pred = self.main_net(states).gather(1, actions)[:, 0]
        with torch.no_grad():
            max_action = self.main_net(next_states).argmax(dim=-1)[:, None]
            max_next_q = self.target_net(next_states).gather(1, max_action)[:, 0]
            # max_next_q = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + (1-dones) * self.gamma * max_next_q
            
            
            
   
            
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.train_step_count += 1
        if self.train_step_count % self.syncnet_freq == 0:
            self.syncNet()
        

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
    gamma = 0.95
    agent = Agent(rows, cols, n_actions, gamma, device)
    n_episode = 4000
    batch_num = 32
    ## 构建replay
    env.reset()
    pbar = tqdm()
    n_train_step = 100000
    visit = np.zeros((rows, cols), dtype=np.int32)
    state_vector = toTensor(env.state_vector).float().to(device)
    for _ in range(10000):
        state = env.pos2state(env.cur_pos)
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        
    done_count = 0
    for transition in agent.memory.buffer:
        if transition[4]:
            done_count += 1
    print(f"{done_count=}")
    
        
        
    for i in range(n_train_step):
        agent.train_step()
        pbar.update(1)
        pbar.set_description(f"episode:{i+1}/{n_episode}")
        if i%1000==0:
            with torch.no_grad():
                policy = agent.target_net(state_vector).argmax(dim=-1).reshape(-1, env.cols)
                render.render_policy(policy)
    # render.render_value(visit)
    agent.syncNet()
    print("训练完成")
    with torch.no_grad():
        policy = agent.target_net(state_vector).argmax(dim=-1).reshape(-1, env.cols)
        render.render_policy(policy)
    

train()

        
        


    