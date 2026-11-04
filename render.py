from env import Env
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.2f}'})
class Render:

    def __init__(self, env:Env):
            self.env = env
    
    def render_policy(self, policy):
        """同时打印 Grid 和 Policy。"""
        symbols = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            4: "·",
        }

        grid = self.env.grid
        rows, cols = grid.shape

        for r in range(rows):
            print("|", end="")

            for c in range(cols):
                cell = grid[r, c]
                action = int(policy[r, c])

                if cell == 1:
                    symbol = " ★ "
                elif cell == -1:
                    symbol = f"■{symbols[action]} "
                else:
                    symbol = f" {symbols[action]} "

                print(f"{symbol}|", end="")

            print()
            print("+" + "---+" * cols)
        
    
    def render_value(self, value):
        grid = self.env.grid
        rows, cols = grid.shape
        for r in range(rows):
            print("|", end="")
            for c in range(cols):
                cell = value[r, c].round(2)
                print(f"{cell:.2f}|", end="")
            print()
            print("+" + "---+" * cols)