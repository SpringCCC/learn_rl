from env import Env
from render import Render
from base_agent import BaseAgent
import numpy as np



class Agent(BaseAgent):

    def __init__(self, rows, cols, n_actions, gamma, env:Env):
        super().__init__(rows, cols, n_actions, gamma)
        self.policy = np.zeros((rows, cols), dtype=np.int32)
        self.v      = np.zeros((rows, cols), dtype=np.float32)
        self.q_table= np.zeros((rows, cols, n_actions), dtype=np.float32)
        self.env = env
        self.truncat_step = 10

    def policy_improve(self):
        self.update_qtable()
        self.policy = self.q_table.argmax(axis=-1)


    def policy_evaluation(self):
        for _ in range(self.truncat_step):
            for r in range(self.rows):
                for c in range(self.cols):
                    a = self.policy[r, c]
                    self.env.cur_pos = (r, c)
                    next_pos, reward, done = self.env.step(a)
                    self.v[r, c] = reward + self.gamma*self.v[next_pos]

    def update_qtable(self):
        for r in range(self.rows):
            for c in range(self.cols):
                for a in range(self.n_actions):
                    self.env.cur_pos = (r, c)
                    next_pos, reward, _ = self.env.step(a)
                    self.q_table[r, c, a] = reward + self.gamma*self.v[next_pos]

    def get_policy(self):
        return self.q_table.argmax(axis=-1)

def policy_iteration():
    env = Env()
    render = Render(env)
    gamma = 0.9
    agent = Agent(env.rows, env.cols, env.n_actions, gamma, env)
    cnt = 0
    while True:
        cnt += 1
        print(f"第{cnt}次迭代policy iteration,策略为:\n")
        old_policy = agent.get_policy()
        agent.policy_evaluation()
        agent.policy_improve()
        if (old_policy!=agent.get_policy()).sum()==0:
            break
        render.render_policy(agent.get_policy())
    render.render_policy(agent.get_policy())


if __name__ == "__main__":
    policy_iteration()