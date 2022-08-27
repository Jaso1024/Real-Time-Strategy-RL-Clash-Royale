from collections import deque
import numpy as np
import time
import os

import matplotlib.pyplot as plt

from CRHandler import Handler
from CRAgent import Agent


class CRBot:
    def get_reward(self, env, done):
        """
        Returns the reward for the latest action/episode.

        :param env: A ClashRoyaleHandler object
        :param duration: A float representing the length of the battle in seconds
        :param done: A boolean representing the episodes completion
        :param action_active: A boolean representing the latest actions impact on the environment
        :return: A Float representing the reward for the latest action/episode
        """

        if done:
            time.sleep(15)  # Wait for game's end animation to finish
            player_crowns = env.get_player_crowns()
            enemy_crowns = env.get_enemy_crowns()

            # rewards for crowns
            # This log function produces a slope that gives less reward
            # crowns to rewards {0:0, 1:10, 2:13, 3:15}
            crowns_reward = np.round(4.96392 * np.log(4.86466 * player_crowns + 0.753851) + 1.40353)
            crowns_reward -= np.round(4.96392 * np.log(4.86466 * enemy_crowns + 0.753851) + 1.40353)

            if player_crowns > enemy_crowns:
                end_reward = 100.0
            elif enemy_crowns > player_crowns:
                end_reward = -100.0
            else:
                end_reward = 0

            total_reward = crowns_reward + end_reward
            return total_reward
        else:
            return 1.0

    def step(self, agent, env, state, duration):
        """
        Executes one action step in the environment.

        :param agent: A BattleAgent object
        :param env: A ClashRoyaleAgent
        :param state: A dictionary containing data about the current state of the Clash Royale window
        :param duration: A float representing the length of the current episode
        :return: A tuple (new_state, reward, done, action, action_active)
        """
 
        

        if duration < 15:
            done = False
        elif duration > 420:
            done = True
            time.sleep(360)
        else:
            done = env.game_is_over()

        actions, probs, vals = agent.act(state)
        reward = self.get_reward(env, done)
        return state, actions, probs, vals, reward, done




