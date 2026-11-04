import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import random
from springc_utils import toNumpy, toTensor

class Env:
    """
    简单的 Grid World 环境。

    网格数值：
        -1：障碍物
         0：普通区域
         1：终点
    
    坐标都是按照[row, col]来表示的，也就是[y, x] [行，列]
    """

    OBSTACLE = -1
    EMPTY = 0
    GOAL = 1
    
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # KEEP = 4
    
    ACTION = {
        UP:     np.asarray([-1, 0]),
        DOWN:   np.asarray([1, 0]),
        LEFT:   np.asarray([0, -1]),
        RIGHT:  np.asarray([0, 1]),
        # KEEP:   np.asarray([0, 0])
    }
    
    
    def __init__(self, rows=6, cols=8, obstacles=None, goal=None, obstacle_rate=0.2):
        """
        Args:
            rows: 网格行数
            cols: 网格列数
            obstacles: 障碍物坐标列表，坐标格式为 [row, col]
            goal: 终点坐标，格式为 (row, col)
        """
        self.obstacle_rate = obstacle_rate
        self.n_actions = len(self.ACTION)
        self.n_states = rows * cols
        self.rows = rows
        self.cols = cols
        self.total_grid = rows * cols
        self.goal = goal if goal is not None else (rows-1, cols-1)
        self.grid = None
        self.obstacles = obstacles if obstacles is not None else self.generate_obstacles()
        self.create_grid()
        self.cur_pos = None
        
        self.obstacle_value = -10
        self.dst_value = 10
        self.step_value = -1
        n_grid = rows * cols
        self.state_vector = np.zeros((n_grid, n_grid), dtype=np.float32)
        self.n_step = 0
        self.max_step = 50
        for i in range(n_grid):
            self.state_vector[i, i] = 1.0
        
        

    def _check_position(self, position):
        """检查坐标是否在网格范围内。"""
        row, col = position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"坐标 {position} 超出网格范围："
                f"行 0~{self.rows - 1}，列 0~{self.cols - 1}"
            )

    def create_grid(self):
        """创建网格，并放置障碍物和终点。"""
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

        # 放置障碍物
        for position in self.obstacles:
            self._check_position(position)
            if position == self.goal:
                raise ValueError("终点不能与障碍物重合")
            self.grid[position] = self.OBSTACLE

        # 放置终点
        self._check_position(self.goal)
        self.grid[self.goal] = self.GOAL

        return self.grid

    def render_env(self):
        symbols = {
            -1: "■",  # 障碍物
            0: "·",  # 普通区域
            1: "★",  # 终点
        }

        print("+" + "---+" * self.cols)

        for row in range(self.rows):
            print("|", end="")

            for col in range(self.cols):
                value = self.grid[row, col]
                print(f" {symbols[value]} |", end="")

            print()
            print("+" + "---+" * self.cols)

    def print_grid(self):
        """在终端打印网格数值。"""
        print(self.grid)

    def reset(self):
        """重新选择初始位置"""
        # while True:
        #     self.cur_pos = (np.random.randint(self.rows), np.random.randint(self.cols))
        #     if self.cur_pos != self.goal:
        #         break
        self.cur_pos = (0, 0)
        self.n_step = 0
    
    
    def generate_obstacles(self):
        """
        生成障碍物。

        Args:
            obstacles: 手动指定的坐标列表；None 表示随机生成。
            obstacle_num: 随机障碍物数量。

        Returns:
            障碍物坐标列表。
        """
        obstacle_num = int(self.total_grid * self.obstacle_rate)
        # 默认排除终点，其他格子都可以成为障碍物
        candidates = [(row, col) for row in range(self.rows) for col in range(self.cols) if (row, col) != self.goal]
        random.shuffle(candidates)
        return candidates[:obstacle_num]
    
    
    def step(self, action):
        self.n_step += 1
        cur_row, cur_col = self.cur_pos
        cur_pos = toNumpy(self.cur_pos)
        new_pos = cur_pos + self.ACTION[action]
        new_row, new_col = new_pos[0].item(), new_pos[1].item()
        if new_row<0 or new_row>=self.rows:
            new_row = cur_row
        if new_col<0 or new_col>=self.cols:
            new_col = cur_col
        next_pos = (new_row, new_col)
        done = False
        if next_pos == self.goal:
            r = self.dst_value
            done = True
        elif next_pos in self.obstacles:
            r = self.obstacle_value
        else:
            r = self.step_value
        self.cur_pos = next_pos
        # if self.n_step > self.max_step:
        #     done = True
        next_state = self.pos2state(next_pos)
        return next_state, r, done
    
    
    def pos2state(self, pos):
        index = pos[0] * self.cols + pos[1]
        return self.state_vector[index]