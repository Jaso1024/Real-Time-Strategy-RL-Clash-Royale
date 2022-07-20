import pandas as pd
import numpy as np

from BattleModelS import BattleModel
from ReplayBuffer import ReplayBuffer

from tensorflow import keras
from keras.optimizers import Adam
import tensorflow as tf
import os
from itertools import repeat
import multiprocessing as multi

def train_model(sample, battle_model, target_model, gamma):
    state, action, state_, reward, done = sample
    target = target_model.predict(state, verbose=0)[0]
    next_state_val = battle_model.predict(state_, verbose=0)[0]
    max_action = np.argmax(battle_model.predict(state_, verbose=0)[0])
    target[action] = reward + gamma * next_state_val[max_action] * done
    battle_model.fit(state, tf.stack([target]), verbose=0)

# Dueling Double Deep Q-learning Agent with an immediate replay buffer
class BattleAgent:
    def __init__(self, mem_size=5000, gamma=.95, epsilon = 0.5, decay=0.95, min_epsilon=0.01, lr=0.001, replace=50, load=True, double=True) -> None:
        if min_epsilon is None:
            min_epsilon = 0.00

        self.battle_model = BattleModel()
        self.target_model = BattleModel() if double else None

        if load:
            checkpoint_num = len(os.listdir("Resources/Models/Saved/BattleModelT4Checkpoints"))
            checkpoint_location = f"Resources/Models/Saved/BattleModelT4Checkpoints/Checkpoint{checkpoint_num}/checkpoint{checkpoint_num}"
            self.battle_model.load_weights(checkpoint_location)
            

        self.memory = ReplayBuffer(mem_size)
        self.gamma = gamma
        self.replace = replace
        self.trainsteps = 1
        self.battle_model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        self.target_model.compile(optimizer=Adam(learning_rate=lr), loss="mse") if double else None
        self.epsilon = epsilon
        self.epsilon_decay = decay
        self.min_epsilon = min_epsilon
    
    def act(self, env, state):
        choices = state["choice_data"]
        if np.random.random() < self.epsilon:
            action = np.random.randint(low=0, high=len(choices))
        else:
            prediction = self.battle_model.advantage(state)[0]
            action = np.argmax(prediction)

        env.act(choices[action])

        if action != 0:
            return action, True
        else:
            return action, False
    
    def update_target(self):
        self.target_model.set_weights(self.battle_model.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.min_epsilon else self.epsilon

    def experience(self, state, action, state_, reward, done):
        self.memory.add((state, action, state_, reward, done)) # We need done to tell the model to not factor in the next state

    def remember(self, size=5):

        samples = self.memory.sample(size)
        if samples is not None:
            for sample in samples:
                state, action, state_, reward, done = sample
                target = self.target_model.predict(state, verbose=0)[0]
                next_state_val = self.battle_model.predict(state_, verbose=0)[0]
                max_action = np.argmax(self.battle_model.predict(state_, verbose=0)[0])
                target[action] = reward + self.gamma * next_state_val[max_action] * done
                self.battle_model.fit(state, tf.stack([target]), verbose=0)
                self.trainsteps += 1

    def remember_multi(self, size = 5):
        if self.trainsteps % self.replace == 0:
            print("Target model updated")
            self.update_target()

        samples = self.memory.sample(size)
        if samples is not None:
            with multi.Pool(5) as p:
                p.starmap(train_model, zip(
                    samples, 
                    repeat(a),
                    repeat(target_model),
                ))

if __name__ == "__main__":
    agent = BattleAgent(load=False)
    print(agent.battle_model.get_weights())
    print(agent.target_model.get_weights())   
