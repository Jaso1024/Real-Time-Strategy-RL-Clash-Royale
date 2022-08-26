import numpy as np
from BattleModel import BattleModel
from ReplayBuffer import ReplayBuffer
from keras.optimizers import Adam
import tensorflow as tf
import os
from keras.callbacks import History 

from ClashRoyaleHandler import ClashRoyaleHandler



class BattleAgent:
    """A Double Dueling Deep-Q-Learning Agent with a Prioritized Replay Buffer"""
    def __init__(self, gamma=.95, epsilon=0.5, decay=0.95, min_epsilon=0.01, lr=0.001, load=False):
        self.battle_model = BattleModel()
        self.target_model = BattleModel()
        self.memory = ReplayBuffer()
        if load:
            checkpoint_num = len(os.listdir("Resources/Models/Saved/BattleModelT1Checkpoints"))
            checkpoint_location = f"Resources/Models/Saved/BattleModelT1Checkpoints/Checkpoint{checkpoint_num}/checkpoint{checkpoint_num}"
            self.battle_model.load_weights(checkpoint_location)
            self.memory = ReplayBuffer(load=load)

        self.battle_model.compile(optimizer="adam", loss="mse")
        self.target_model.compile(optimizer="adam", loss="mse")
        
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.origin_square_locations = self.get_origin_square_locations()
    
    def get_origin_square_locations(self):
        locations = []
        for x in range(1, 18-1, 2):
            for y in range(1, 14-1, 2):
                locations.append((x,y))
        return locations
    
    def get_tile(self, tile_of_nine):
        tile_mappings = {1:(-1,-1), 2:(-1,0), 3:(-1,1),
                         4:(0,-1), 5:(0,0), 6:(0,1),
                         7:(1,-1), 8:(1,0), 9:(1,1)}
        return tile_mappings[tile_of_nine]

    def make_action(self, tile, card):
        if type(tile) == dict:
            action = {}
            for key, value in tile.items():
                if key == 'card':
                    continue
                else:
                    action[key] = value
            action['card'] = card
        else:
            action = tile
        return action
    
    def get_action(self, action_components, choices, card_data):
        action = {}
        origin_squares_data = []
        tile_matrix = self.to_matrix(choices)
        for x, y in self.origin_square_locations:
            origin_squares_data.append([x, y])
        origin_tile_location = origin_squares_data[action_components[0]]
        tile_component = action_components[1]
        tile_component = self.get_tile(tile_component)
        tile_location = (origin_tile_location[0] + tile_component[0], origin_tile_location[1] + tile_component[1])
        tile = tile_matrix[tile_location]
        action = self.make_action(tile, card_data[action_components[2]])
        return action
        
    def to_matrix(self, choices):
        current_idx = 0
        choice_matrix = []
        for x in range(18):
            choice_vector = []
            for y in range(14):
                choice_vector.append(choices[current_idx])
                current_idx += 4
            choice_matrix.append(choice_vector)
        return np.array(choice_matrix)

    def act(self, env, state, memories):
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
            action_components = self.battle_model.advantage(state)
            if None in action_components:
                action = action_components[0]
            else:
                action = self.get_action(action_components, choices, state['card_data'])

        remembered = False 
        if memories > 0:
            self.remember(memories)
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
    agent = BattleAgent()
    env = ClashRoyaleHandler()
    state = env.get_state()
    agent.act(env, state, memories=0)
