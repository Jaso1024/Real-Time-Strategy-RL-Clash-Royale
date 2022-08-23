from subprocess import call
import numpy as np
from BattleModel import BattleModel
from ReplayBuffer import ReplayBuffer
from keras.optimizers import Adam
import tensorflow as tf
import os
from keras.callbacks import History 



class BattleAgent:
    """A Double Dueling Deep-Q-Learning Agent with a Prioritized Replay Buffer"""
    def __init__(self, gamma=.95, epsilon=0.5, decay=0.95, min_epsilon=0.01, lr=0.001, load=True):
        self.battle_model = BattleModel()
        self.target_model = BattleModel()
        self.memory = ReplayBuffer()
        if load:
            checkpoint_num = len(os.listdir("Resources/Models/Saved/BattleModelT1Checkpoints"))
            checkpoint_location = f"Resources/Models/Saved/BattleModelT1Checkpoints/Checkpoint{checkpoint_num}/checkpoint{checkpoint_num}"
            self.battle_model.load_weights(checkpoint_location)
            self.memory = ReplayBuffer(load=True)

        self.battle_model.compile(optimizer="adam", loss="mse")
        self.target_model.compile(optimizer="adam", loss="mse")
        
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        self.epsilon = epsilon
        self.gamma = gamma
    
    def act(self, env, state):
        """
        Executes an action.

        :param env: A ClashRoyaleHandler object
        :param state: A dictionary representing the current state of the Clash Royale window
        :return: A tuple (int - the action, boolean - if the action change the environment)
        """
        choices = state["choice_data"]
        if np.random.random() < self.epsilon:
            action = np.random.randint(low=0, high=len(choices))
        else:
            prediction = self.battle_model.advantage(state)[0]
            action = np.argmax(prediction)
        remembered = False
        if all(choices[choice]==None for choice in range(1,len(choices))):
            self.remember(5)
            remembered = True
        env.act(choices[action])
        if choices[action] is None:
            return action, False, remembered
        else:
            return action, True, remembered

    def update_target(self):
        """Updates target network."""
        self.target_model.set_weights(self.battle_model.get_weights())

    def update_epsilon(self):
        """Updates the epsilon."""
        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.min_epsilon else self.epsilon

    def experience(self, state, action, state_, reward, done, loss=None):
        """Adds sample to replay buffer."""
        if loss is None:
            loss = self.remember_one((state, action, state_, reward, done))
        self.memory.add((state, action, state_, reward, done), loss)
    
    def save(self):
        """Saves the replay buffer's samples."""
        self.memory.save()

    def forget(self):
        """removes samples from the replay buffer."""
        self.memory.remove()

    def remember(self, size=5):
        """
        Trains the agent.

        :param size: An integer representing how many samples should be used in training
        :return: None
        """
        samples = self.memory.sample(size)
        if samples is not None:
            for sample in samples:
                state, action, state_, reward, done = sample
                loss = self.remember_one((state,action,state_,reward,done))
                self.experience(state,action,state_,reward,done,loss)

    def remember_one(self, experience):
        history = History()
        state, action, state_, reward, done = experience
        target = self.target_model.predict(state, verbose=0)[0]
        next_state_val = self.battle_model.predict(state_, verbose=0)[0]
        max_action = np.argmax(self.battle_model.predict(state_, verbose=0)[0])
        target[int(action)] = reward + self.gamma * next_state_val[max_action] * (not done)
        self.battle_model.fit(state, tf.stack([target]), verbose=0, callbacks=[history])
        return history.history['loss'][0]

if __name__ == "__main__":
    agent = BattleAgent(load=True)
    agent.remember(180)
    agent.save()
