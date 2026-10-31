from collections import deque
import numpy as np
import random
from springc_utils import toTensor, toNumpy



class ReplayBuffer():
    
    def __init__(self, max_buffer_num, batch_size, device=None):
        self.buffer = deque(maxlen=max_buffer_num)
        self.device = device
        self.batch_size = batch_size
        
    def push(
        self,
        state:np.ndarray,
        action:int,
        reward:float,
        next_state:np.ndarray,
        done:bool
        ) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self):
        data = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*data)
        state = toTensor(state, self.device).float()
        action = toTensor(action, self.device).long()[:, None]
        reward = toTensor(reward, self.device).float()
        next_state = toTensor(next_state, self.device).float()
        done = toTensor(done, self.device).float()
        return state, action, reward, next_state, done
        
        
        
    def __len__(self):
        return len(self.buffer)