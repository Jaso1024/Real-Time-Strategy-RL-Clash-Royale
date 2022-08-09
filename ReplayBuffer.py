from collections import deque
import numpy as np
import random
import pandas as pd


class ReplayBuffer:
    """A Combined Replay Buffer with priorities."""

    def __init__(self, load=False):
        self.memory = deque()
        self.recents = deque()
        self.length = 10000
        self.losses = deque()
        self.memoryframe = pd.DataFrame({"state": [], "action": [], "state_": [], "reward": [], "done": []})
        
        if load:
            mem1 = pd.read_pickle("Resources/Memories/mem1.pkl")
            mem2 = pd.read_pickle("Resources/Memories/mem2.pkl")
            mem3 = pd.read_pickle("Resources/Memories/mem3.pkl")
            self.memoryframe = pd.concat([mem1, mem2, mem3])
            for entry in self.memoryframe.values.tolist():
                self.memory.appendleft(entry)
            self.losses = deque(list(np.genfromtxt("Resources/memories/losses.csv", delimiter=',')))

    def add(self, experience, loss):
        """Adds an experience to a replay buffer."""
        self.memory.appendleft(experience)
        self.losses.appendleft(loss)
        self.recents.appendleft(experience)

    def save(self):
        """Saves all experiences in the replay buffer to csv files."""

        states = []
        actions = []
        states_ = []
        rewards = []
        dones = []
        for memory in self.memory:
            state, action, state_, reward, done = memory
            states.append(state)
            actions.append(action)
            states_.append(state_)
            rewards.append(reward)
            dones.append(done)

        frame_to_save = pd.DataFrame({"state": states, "action": actions, "state_": states, "reward": rewards, "done": dones})
        self.split_save(frame_to_save)

    def split_save(self, frame):
        """Split dataframe into 3, then save individually"""
        frame_length = len(frame)
        frame.iloc[:int(frame_length / 3)].to_pickle("Resources/Memories/mem1.pkl")
        frame.iloc[int(frame_length / 3):int(((frame_length / 3) + (frame_length / 3)))].to_pickle(
            "Resources/Memories/mem2.pkl")
        frame.iloc[int(((frame_length / 3) + (frame_length / 3))):].to_pickle("Resources/Memories/mem3.pkl")

    def sample(self, batch_size=5):
        """
        Generates a sample of experiences from the replay buffer.

        :param batch_size: The amount of experiences to sample
        :return: A generator object which yields singular memories
        """
        losses = np.array(self.losses)
        priorities = losses / max(losses)
        sample_indicies = random.choices(range(len(self.memory)), k=batch_size - 1, cum_weights=priorities)

        for idx in sample_indicies:
            experience = self.memory[idx][:]
            del self.memory[idx]
            del self.losses[idx]
            yield experience

    def remove(self):
        """Removes extra memories from the replay buffer."""
        while len(self.memory) > self.length:
            try:
                idx = random.choice([num for num in range(len(self.memory))])
                del self.memory[idx]
                del self.losses[idx]
            except Exception as e:
                print(e)
                continue
