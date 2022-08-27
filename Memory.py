import os
import numpy as np
import tensorflow as tf
from collections import deque
import random

class Memory:
    def __init__(self, batch_size) -> None:
        self.mem
        self.batch_size = batch_size

    def generate_batches(self):
        sample_indicies = random.choices(range(len(self.memory)), k=self.batch_size)
        for idx in sample_indicies:
            experience = self.mem[idx][:]
            yield experience
        
    def store(self, state, action, probs, vals, reward, done):
        self.mem.appendleft([state, action, probs, vals, reward, done])
    
    def clear(self):
        self.mem = deque()