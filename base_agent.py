import numpy as np 


class BaseAgent:
    def __init__(self, rows, cols, n_actions, gamma):
        self.n_actions = n_actions
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        # self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float32) ##
        self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float32) ##
        