
from env import Env
import numpy as np
from render import Render
import random
random.seed(143)
from torch.utils.tensorboard import SummaryWriter

class Sarsa():
    
    def __init__(self, 
                 rows, cols, n_actions,
                 alpha=0.1, 
                 gamma=0.9, 
                 epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rows = rows
        self.cols = cols
        self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float32) ##
        
    def select_action(self, state):
        if np.random.random()<self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            q_values = self.q_table[state]
            best_actions = np.flatnonzero(q_values == q_values.max())
            action =  np.random.choice(best_actions).item()
        return action
        
    
    def update(self, s0, a0, r1, s1, a1, done):
        if done:
            td_target = r1
        else:
            td_target = r1 + self.gamma * self.q_table[s1][a1]
        td_error = td_target - self.q_table[s0][a0]
        self.q_table[s0][a0] += self.alpha * td_error

    def get_policy(self):
        return np.argmax(self.q_table, axis=2)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def calc_epsilon(i_episode, total_episode, max_epsilon=0.5, min_epsilon=0.1):
    return min_epsilon + (max_epsilon - min_epsilon) * (total_episode - i_episode) / total_episode

def train():
    writer = SummaryWriter(log_dir="runs/vbbs")
    env = Env()
    render = Render(env)
    agent = Sarsa(env.rows, env.cols, env.n_actions)
    n_episode = 5000
    max_epsilon = 0.5
    min_epsilon = 0.05

    for i in range(n_episode):
        epsilon = calc_epsilon(i, n_episode/2, max_epsilon, min_epsilon)
        agent.set_epsilon(epsilon)
        writer.add_scalar("Parameters/epsilon", epsilon, i)
        env.reset()
        s0 = env.cur_pos
        a0 = agent.select_action(s0)
        ce = 0
        episode_return = 0
        while True:
            ce += 1
            s1, r, done = env.step(a0)
            episode_return += r
            a1 = agent.select_action(s1)
            agent.update(s0, a0, r, s1, a1, done)
            s0 = s1
            a0 = a1
            if done or ce>1000:
                break
        writer.add_scalar("Return/Episode", episode_return, i)
        if i==(n_episode-2):
            ttttt= 1
        if i>0 and i%500==0:
            print(f"Episode {i} finished.")
            render.render_policy(agent.get_policy())    
    writer.close()


train()
