from env import Env
from render import Render
from base_agent import BaseAgent
import numpy as np
import random
from tqdm import tqdm


class Agent(BaseAgent):

    def __init__(self, rows, cols, n_actions, gamma, on_policy):
        super().__init__(rows, cols, n_actions, gamma)
        self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float32)
        self.epsilon = 0.1
        self.alpha = 0.1
        self.on_policy = on_policy


    def select_action(self, state):
        if self.on_policy:
            if np.random.random() < self.epsilon:
                select_a =  np.random.randint(self.n_actions)
            else:
                qv = self.q_table[state]
                index = np.flatnonzero(qv==qv.max())
                select_a =  np.random.choice(index).item()
        else:
            select_a = np.random.randint(self.n_actions)
        return select_a

    def update_q(self, s0, a0, r1, s1):
        q_value = self.q_table[s0][a0]
        td_target = r1 + self.gamma * max(self.q_table[s1])
        td_error = td_target - q_value
        self.q_table[s0][a0] += self.alpha * td_error

    def get_policy(self):
        return self.q_table.argmax(axis=-1)



def train():
    env = Env()
    render = Render(env)
    gamma = 0.9
    rows, cols, n_actions = env.rows, env.cols, env.n_actions
    on_policy = False
    agent = Agent(rows, cols, n_actions, gamma, on_policy)
    n_episode = 10
    steps_episode = 100000
    env.reset()
    visit = np.zeros((rows, cols), dtype=np.int32)


    for k in range(n_episode):
        old_policy = agent.get_policy()
        env.reset()
        pbar = tqdm(total=steps_episode)
        s0 = env.cur_pos
        a0 = agent.select_action(s0)
        visit *= 0
        for i in range(steps_episode):
            s1, reward, done = env.step(a0)
            agent.update_q(s0, a0, reward, s1)
            s0 = s1
            a0 = agent.select_action(s0)
            visit[s1] += 1

            pbar.update(1)
            pbar.set_description(f"Epoch {i + 1}")

        print(f"第{k+1}/{n_episode}条episode探索完成, 策略为")
        render.render_policy(agent.get_policy())
        render.render_value(visit)
        new_policy = agent.get_policy()
        if (old_policy!=new_policy).sum()==0:
            print("策略已经没有提升，提前结束迭代")
            break

if __name__ == "__main__":
    train()