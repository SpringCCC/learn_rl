from env import Env
import numpy as np
from base_agent import BaseAgent
from render import Render

class Agent(BaseAgent):

    def __init__(self, rows, cols, n_actions, gamma, env:Env):
        super().__init__(rows, cols, n_actions, gamma)
        self.v = np.zeros((rows, cols), dtype=np.float32)  # 初始化状态值函数
        self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float32)  # 初始化动作值函数
        self.env = env

    def update_v(self):
        self.v = self.q_table.max(axis=2)  # 更新状态值函数为动作值函数的最大值

    def update_q(self):
        for r in range(self.rows):
            for c in range(self.cols):
                for a in range(self.n_actions):
                    self.env.cur_pos = (r, c)
                    next_pos, reward, _ = self.env.step(a)
                    self.q_table[r, c, a] = reward + self.gamma * self.v[next_pos]


    def get_policy(self):
        return np.argmax(self.q_table, axis=2)


def value_iteration():
    env = Env()
    render = Render(env)
    gamma = 0.9
    rows, cols = env.rows, env.cols
    n_actions = env.n_actions
    agent = Agent(rows, cols, n_actions, gamma, env)
    error = 0.01
    cnt = 0
    while True:
        cnt += 1
        print(f"第{cnt}次更新v和q")
        agent.update_q()
        old_v = agent.v.copy()
        agent.update_v()
        diff = abs(old_v - agent.v)
        if (diff>error).sum()==0:
            break
        if cnt %10==0:
            render.render_policy(agent.get_policy())
    print("最终的Policy")
    policy = agent.get_policy()
    render.render_policy(policy)


if __name__ == "__main__":
    value_iteration()

    




       