from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, size=2000):
        self.buffer = deque()
        self.priorities = deque()
        self.length = size
    
    def add(self, experience):
        self.buffer.appendleft(experience)
        self.priorities.appendleft(max(self.priorities, default=1))
        self.check_mem()
    
    def get_probs(self, scale):
        scaled_priorities = np.array(self.priorities * scale)
        probabilities = scaled_priorities/sum(scaled_priorities)
        return probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance = importance / max(importance)
        return importance

    def check_mem(self):
        if len(self.buffer) > self.length:
            sample_indicies = random.choices(range(len(self.buffer)), k=4)
            for num in sample_indicies:
                del self.buffer[num]

    def sample(self, batch_size = 5, scale = 1.0):
        if batch_size > len(self.buffer):
            return None
        else:
            #probabilities = self.get_probs(scale)
            sample_indicies = random.choices(range(len(self.buffer)), k=batch_size-1)
            samples = np.array(self.buffer)[[0, *sample_indicies]]
            #importance = self.get_importance(probabilities[sample_indicies])
            return samples
        