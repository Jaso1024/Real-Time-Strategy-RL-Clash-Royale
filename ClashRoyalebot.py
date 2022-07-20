from ClashRoyaleHandler import ClashRoyaleHandler
from BattleModel import BattleModel
from BattleAgent import BattleAgent

import numpy as np
import pandas as pd
import time
import os

import matplotlib.pyplot as plt

class ClashRoyaleBot:
    def __init__(self):
        self.ep_record = []



    def get_reward(self, env, duration, active_actions_made):
        def get_actions_reward(duration, active_actions_made):
                possible_actions = 5
                if duration <= 120.0:
                    possible_actions += duration/2.8
                elif duration <= 180.0:
                    possible_actions += 120.0/2.8 + (duration-120.0)/1.4
                elif duration <= 420:
                    possible_actions += 120.0/2.8 + 60.0/1.4 + ((duration-180.0)/1.4)

                num_actions_reward = 1-(active_actions_made/possible_actions)**0.5
                return num_actions_reward
        
        if duration > 420:
            return 0

        if env.game_is_over():
            actions_reward = get_actions_reward(duration, active_actions_made)
            time.sleep(15) # Wait for end game animation to finish

            player_crowns = env.get_player_crowns()
            enemy_crowns = env.get_enemy_crowns()

            # rewards for crowns
            # This log function produces a slope that gives less reward
            # for every crown after the first
            crowns_reward = np.round(4.96392*np.log(4.86466*player_crowns+0.753851)+1.40353)
            crowns_reward -= np.round( 4.96392*np.log(4.86466*enemy_crowns+0.753851)+1.40353)

            if player_crowns>enemy_crowns:
                end_reward = 100
            elif enemy_crowns>player_crowns:
                end_reward = -100
            else:
                end_reward = 0

            total_reward = actions_reward + crowns_reward + end_reward
            return total_reward
        else:
            return 1 # Reward for not winning or losing

    def step(self, agent, env, state, duration, active_actions):
        assert isinstance(agent, BattleAgent)
        action, action_active = agent.act(env, state)
        done = env.game_is_over() if duration > 15 else False
        return env.get_state(), self.get_reward(env, duration, active_actions), done, action, action_active

    def record_episode(self, result, epsilon):
        self.ep_record.append((len(self.ep_record), epsilon, result))

    def scatter(self):
        ep_nums, epsilons, results = list(map(list, zip(*self.ep_record)))
        plt.plot(ep_nums, results)
        plt.show()
        plt.plot(epsilons, results)
        plt.show()

    def start_battle(self, env):
        env.ignore_new_reward()
        time.sleep(1)
        env.battle()
        if env.check_reward_limit_reached():
            env.acknowledge_reward_limit_reached()
        while env.game_is_over():
            continue

    @staticmethod
    def leave_game(env):
        wait_start_time = time.time()
        while not env.at_home_screen():
            if time.time() - wait_start_time > 10:
                break
            env.leave_game()
            time.sleep(5)

    def checkpoint_model(self, agent):
        checkpoint_num = len(os.listdir("Resources/Models/Saved/BattleModelT4Checkpoints"))+1
        os.mkdir(f"Resources/Models/Saved/BattleModelT4Checkpoints/Checkpoint{checkpoint_num}")
        agent.battle_model.save_weights(f"Resources/Models/Saved/BattleModelT4Checkpoints/Checkpoint{checkpoint_num}/checkpoint{checkpoint_num}", save_format = "tf")

    def print_episode_stats(self, ep_num, duration, epsilon, actions, active_actions, reward, rewards):
        print(f"Episode: {ep_num} | Episode duration: {duration} | Epsilon: {epsilon} | Actions chosen: {actions} | Active Actions: {active_actions} | Reward: {reward} | Max Reward: {max(rewards)}")
        print("-------------------------------------------------------------------------------------------------------")

    def run_episode(self, agent, env, ep_num, learn=True):
        done = False
        active_actions = 0
        total_actions = 0
        total_reward = 0
        rewards = []
        assert isinstance(env, ClashRoyaleHandler)
        self.start_battle(env)
        state = env.get_state()

        episode_start_time = time.time()
        while not done:
            duration = time.time() - episode_start_time

            new_state, reward, done, action, action_active = self.step(agent, env, state, duration, active_actions)

            total_reward += reward
            total_actions += 1
            rewards.append(reward)
            if action_active:
                active_actions += 1

            if learn:
                agent.experience(state, action, new_state, reward, done)
                agent.remember(5)

            state = new_state

        self.leave_game(env)
        return duration, active_actions, total_actions, total_reward, rewards


    def play(self, episodes=300, learn=True, epsilon=0.5, decay=.9995):
        env = ClashRoyaleHandler()
        agent = BattleAgent(load=True, epsilon=epsilon, decay=decay, replace=10, mem_size=10000) if learn else BattleAgent(epsilon=0.0, load=True)
        for ep in range(1, episodes+1):
            duration, active_actions, total_actions, total_reward, rewards = self.run_episode(agent, env, ep, learn)
            if ep % 25 == 0:
                self.checkpoint_model(agent)
            self.print_episode_stats(ep, duration, agent.epsilon, total_actions, active_actions, total_reward, rewards)
            self.record_episode(total_reward, agent.epsilon)
            agent.update_epsilon()
            agent.remember(180)
            if ep > 1:
                agent.update_target()

        self.scatter()


if __name__ == "__main__":
    bot = ClashRoyaleBot()
    bot.play(500, epsilon=0.4862, decay=0.999)
