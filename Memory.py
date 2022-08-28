import os
import numpy as np
import tensorflow as tf
from collections import deque
import random

class Memory:
    def __init__(self) -> None:
        self.mem = deque()

    def generate_batches(self):
        for idx in range(len(self.mem)):
            experience = self.mem[idx][:]
            yield experience
        
    def store(self, state, action, probs, vals, reward, done):
        self.mem.appendleft([state, action, probs, vals, reward, done])
    
    def clear(self):
        self.mem = deque()