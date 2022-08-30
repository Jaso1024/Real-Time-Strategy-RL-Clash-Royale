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
                end_reward = 1000.0
            elif enemy_crowns > player_crowns:
                end_reward = -1000.0
            else:
                end_reward = 0

            total_reward = crowns_reward + end_reward
            return total_reward
        else:
            return -1.0

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

        actions, probs, vals = agent.act(env, state)
        reward = self.get_reward(env, done)
        new_state = env.get_state()
        return new_state, actions, probs, vals, reward, done

    def start_battle(self, env):
        """Start a competetive battle."""
        env.ignore_new_reward()
        time.sleep(5)
        env.battle()
        time.sleep(3)
        if env.check_reward_limit_reached():
            env.acknowledge_reward_limit_reached()
        while env.game_is_over():
            continue

    @staticmethod
    def leave_game(env):
        """Leave a Clash Royale game."""
        wait_start_time = time.time()
        while not env.at_home_screen():
            env.leave_game()
            time.sleep(10)
    
    def print_episode_stats(self, ep_num, duration, reward):
        """Print episode stats."""
        print(f"Episode: {ep_num} | Duration: {duration} | reward: {reward}")
        print("------------------------------------------------------------")

        
    def run_episode(self, agent, env, learn=True, memories_per_step=0):
        """Runs a single episode."""
        done = False
        total_reward = 0

        self.start_battle(env)
        state = env.get_state()
        episode_start_time = time.time()
        while not done:
            duration = time.time() - episode_start_time
            new_state, actions, probs, vals, reward, done = self.step(agent, env, state, duration)
            agent.act(env, state)
            agent.experience((state, actions, probs, vals, reward, done))
            total_reward += reward
            state = new_state

        print("done")
        self.leave_game(env)
        return duration, total_reward

    def play(self, episodes=300, learn=True, spells=False, epsilon=0.5, decay=.9995, checkpoint=100, remembrance_steps=320, target_episodes=5, scatter=False, load=True):
        """
        Plays a series of Clash Royale game.

        :param episodes: An integer representing the amount of episodes to play
        :param learn: A boolean that limits the Agent's capacity to learn
        :param spells: A boolean representing the spells in the current deck
        :param epsilon: A float that determines the random decision boundary
        :param decay: A float that determines how fast the epsilon decreases
        :param checkpoint: An integer that determines how often the agent will make checkpoints
        :param remembrance_steps: An integer that determines how many state transitions the agent will train with each episode
        :param target_episodes: An integer that determines how often the target network will be updated
        :param scatter: A boolean that indicates that a scatterplot should be shown upon completion of all episodes
        :param load: A boolean that indicates that a previous checkpoint should be loaded
        :return: None
        """
        env = Handler(spells)
        agent = Agent(load)

        episodes = 1000
        best_reward = -10000
        
        for ep in range(1, episodes + 1):
            duration, total_reward = self.run_episode(agent, env, learn)
            if duration < 25:
                break
            self.print_episode_stats(ep, duration, total_reward)
            agent.train()
            if total_reward > best_reward:
                agent.save()

            
if __name__ == '__main__':
    bot = CRBot()
    bot.play(load=False)



