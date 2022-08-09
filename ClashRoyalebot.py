from tracemalloc import start
from ClashRoyaleHandler import ClashRoyaleHandler
from BattleAgent import BattleAgent

import numpy as np
import time
import os
from collections import deque

import matplotlib.pyplot as plt


class ClashRoyaleBot:
    def __init__(self, print_stats=True, record_stats=True):
        self.ep_record = []
        self.print_stats = print_stats
        self.record_stats = record_stats

    def get_reward(self, env, duration, done, action_active):
        """
        Returns the reward for the latest action/episode.

        :param env: A ClashRoyaleHandler object
        :param duration: A float representing the length of the battle in seconds
        :param done: A boolean representing the episodes completion
        :param action_active: A boolean representing the latest actions impact on the environment
        :return: A Float representing the reward for the latest action/episode
        """

        if duration > 420:
            return 0.0
        elif done:
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
            if action_active:
                return 0.5
            else:
                return 0.0

    def step(self, agent, env, state, duration):
        """
        Executes one action step in the environment.

        :param agent: A BattleAgent object
        :param env: A ClashRoyaleAgent
        :param state: A dictionary containing data about the current state of the Clash Royale window
        :param duration: A float representing the length of the current episode
        :return: A tuple (new_state, reward, done, action, action_active)
        """
        assert isinstance(agent, BattleAgent)
        assert isinstance(env, ClashRoyaleHandler)
        action, action_active = agent.act(env, state)

        if duration < 15:
            done = False
        elif duration > 420:
            done = True
            time.sleep(360)
        else:
            done = env.game_is_over()

        return env.get_state(), self.get_reward(env, duration, done, action_active), done, action, action_active

    def record_episode(self, result, epsilon):
        """Record the episode to be plotted later."""
        self.ep_record.append((len(self.ep_record), epsilon, result))

    def scatter(self):
        """Create 2 scatter plots from the bots performance."""
        if self.record_stats:
            ep_nums, epsilons, results = list(map(list, zip(*self.ep_record)))
            plt.plot(ep_nums, results)
            plt.show()
            plt.plot(epsilons, results)
            plt.show()
        else:
            print("Cannot create scatterplot as record_stats is set to False")

    def start_battle(self, env):
        """Start a competetive battle."""
        env.ignore_new_reward()
        time.sleep(1)
        env.battle()
        time.sleep(1)
        if env.check_reward_limit_reached():
            env.acknowledge_reward_limit_reached()
        timer_start = time.time()
        while env.game_is_over():
            continue

    @staticmethod
    def leave_game(env):
        """Leave a Clash Royale game."""
        wait_start_time = time.time()
        while not env.at_home_screen():
            env.leave_game()
            time.sleep(5)

    def checkpoint_model(self, agent):
        """Checkpoint the model for loading later."""
        checkpoint_num = len(os.listdir("Resources/Models/Saved/BattleModelT6Checkpoints")) + 1
        os.mkdir(f"Resources/Models/Saved/BattleModelT6Checkpoints/Checkpoint{checkpoint_num}")
        agent.battle_model.save_weights(
            f"Resources/Models/Saved/BattleModelT6Checkpoints/Checkpoint{checkpoint_num}/checkpoint{checkpoint_num}",
            save_format="tf")

    def print_episode_stats(self, ep_num, duration, epsilon, actions, active_actions, reward, rewards):
        """Print episode stats."""
        print(
            f"Episode: {ep_num} | Episode duration: {duration} | Epsilon: {epsilon} | Actions chosen: {actions} | Active Actions: {active_actions} | Reward: {reward} | Max Reward: {max(rewards)}")
        print("-------------------------------------------------------------------------------------------------------")

    def run_episode(self, agent, env, learn=True):
        """Runs a single episode."""
        done = False
        active_actions = 0
        total_actions = 0
        total_reward = 0
        rewards = []
        dones = deque()

        assert isinstance(env, ClashRoyaleHandler)
        assert isinstance(agent, BattleAgent)
        self.start_battle(env)

        state = env.get_state()

        episode_start_time = time.time()
        while not done:
            duration = time.time() - episode_start_time

            new_state, reward, done, action, action_active = self.step(agent, env, state, duration)

            total_reward += reward
            total_actions += 1
            rewards.append(reward)
            dones.appendleft(done)
            if action_active:
                active_actions += 1

            if learn:
                agent.experience(state, action, new_state, reward, done)

            state = new_state

        self.leave_game(env)
        return duration, active_actions, total_actions, total_reward, rewards

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
        env = ClashRoyaleHandler(spells)

        if learn:
            agent = BattleAgent(load=load, epsilon=epsilon, decay=decay)
        else:
            agent = BattleAgent(epsilon=0.0, load=load)

        for ep in range(1, episodes + 1):
            duration, active_actions, total_actions, total_reward, rewards = self.run_episode(agent, env, learn)
            agent.forget()
            if ep % checkpoint == 0:
                try:
                    self.checkpoint_model(agent)
                except Exception as e:
                    print(f"Failed Checkpoint - Error: {e}")
                    continue
                agent.save()

            if self.print_stats:
                self.print_episode_stats(ep, duration, agent.epsilon, total_actions, active_actions, total_reward, rewards)
            if self.record_stats:
                self.record_episode(total_reward, agent.epsilon)
            agent.update_epsilon()
            start_time = time.time()
            agent.remember(remembrance_steps)
            print(time.time()-start_time)

            if ep > 1 and ep % target_episodes == 0:
                agent.update_target()
        if scatter:
            self.scatter()


if __name__ == "__main__":
    bot = ClashRoyaleBot()
    bot.play(2000, epsilon=.53382, decay=0.9995, scatter=True, load=False)
