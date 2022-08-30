from re import S
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import os
from keras.callbacks import History 

from CRHandler import Handler
from CRModel import StateEncoder, Critic, OriginActor, TileActor, CardActor
from ActionMapper import ActionMapper
from Memory import Memory

class Agent():
    """A Proximal Policy Gradient Agent"""
    def __init__(self, origin_lr=1e-6, shell_lr=1e-5, card_lr=1e-5, gamma=0.95, lam=0.95, clip=0.2, epochs=2, load=False) -> None:
        self.state_encoder = StateEncoder()

        self.origin_actor = OriginActor()
        self.origin_critic = Critic()
        self.shell_actor = TileActor()
        self.shell_critic = Critic()
        self.card_actor = CardActor()
        self.card_critic = Critic()
        
        self.compile(origin_lr, shell_lr, card_lr)

        if load:
            self.load()

        self.action_mapper = ActionMapper()
        self.mem = Memory()

        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epohs = epochs
    
    def compile(self, origin_lr, shell_lr, card_lr):
        self.origin_actor.compile(optimizer=Adam(learning_rate=origin_lr))
        self.origin_critic.compile(optimizer=Adam(learning_rate=origin_lr))
        self.shell_actor.compile(optimizer=Adam(learning_rate=shell_lr))
        self.shell_critic.compile(optimizer=Adam(learning_rate=shell_lr))
        self.card_actor.compile(optimizer=Adam(learning_rate=card_lr))
        self.card_critic.compile(optimizer=Adam(learning_rate=card_lr))
        
    def save(self):
        self.state_encoder.save_weights("state_encoder")
        self.origin_actor.save_weights("origin_actor")
        self.origin_critic.save_weights("origin_critic")
        self.shell_actor.save_weights("shell_actor")
        self.shell_critic.save_weights("shell_critic")
        self.card_actor.save_weights("card_actor")
        self.card_critic.save_weights("card_critic")
    
    def load(self):
        self.state_encoder.load_weights("state_encoder")
        self.origin_actor.load_weights("origin_actor")
        self.origin_critic.load_weights("origin_critic")
        self.shell_actor.load_weights("shell_actor")
        self.shell_critic.load_weights("shell_critic")
        self.card_actor.load_weights("card_actor")
        self.card_critic.load_weights("card_critic")

    def experience(self, experience):
        self.mem.store(*experience)

    def get_action(self, action_components, choices, cards):
        return self.action_mapper.get_action(action_components, choices, cards)
    
    def get_origin_action(self, encoded_state):
        origin_probs = self.origin_actor(encoded_state)
        origin_dist = tf.compat.v1.distributions.Categorical(probs=origin_probs, dtype=tf.float32)
        origin = origin_dist.sample()

        value = self.origin_critic(encoded_state)

        o_prob = origin_dist.log_prob(origin)

        return origin, value, o_prob
    
    def get_shell_action(self, origin):
        origin = tf.squeeze(origin)
        origin = int(tf.get_static_value(origin))
        encoded_origin = np.identity(49)[origin:origin+1]
        shell_probs = self.shell_actor(encoded_origin)
        shell_dist = tf.compat.v1.distributions.Categorical(probs=shell_probs, dtype=tf.float32)
        shell = shell_dist.sample()

        value = self.shell_critic(encoded_origin)

        s_prob = shell_dist.log_prob(shell)

        return shell, value, s_prob

    def get_card_action(self, encoded_state):
        card_probs = self.card_actor(encoded_state)
        card_dist = tf.compat.v1.distributions.Categorical(probs=card_probs, dtype=tf.float32)
        card = card_dist.sample()
        value = self.card_critic(encoded_state)

        c_prob = card_dist.log_prob(card)

        return card, value, c_prob
        

    def act(self, env, state):
        """
        Executes an action.

        :param env: A ClashRoyaleHandler object
        :param state: A dictionary representing the current state of the Clash Royale window
        :return: A tuple (int - the action, boolean - if the action change the environment)
        """
        encoded_state = self.state_encoder(state)
        origin, origin_val, origin_prob = self.get_origin_action(encoded_state)
        shell, shell_val, shell_prob = self.get_shell_action(origin)
        card, card_val, card_prob = self.get_card_action(encoded_state)

        action_components = (origin, shell, card)
        choices = state["choice_data"]
        cards = state['card_data']
        action = self.get_action(action_components, choices, cards)
        env.act(action)

        return (origin, shell, card), (origin_prob, shell_prob, card_prob), (origin_val, shell_val, card_val)
    
    def get_adv(self, values, rewards, dones):
        g = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (1 if not dones[i] else values[i + 1]) * dones[i] - values[i]
            g = delta + self.gamma * self.lam * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        advantage = np.array(returns, dtype=np.float32) - values[:]
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-10)
        return advantage, returns

    def get_adv_vals(self, batches):
        origin_vals = []
        shell_vals = []
        card_vals = []
        rewards = []
        dones = []
        for elem in batches:
            vals = elem[3]
            origin_vals.append(vals[0])
            shell_vals.append(vals[1])
            card_vals.append(vals[2])
            rewards.append(elem[4])
            dones.append(elem[-1])
        return origin_vals, shell_vals, card_vals, rewards, dones

    def get_loss(self, actor, critic, batches, batch, advantage, ret, agent_num, shell=False):
        state, actions, old_probs, vals, reward, done = batches[batch]
        old_probs = old_probs[agent_num]
        action = actions[agent_num]
        vals = vals[agent_num]  
        
        state = self.state_encoder(state)
        if shell:
            state = self.origin_actor(state)
        probs = actor(state)
        dist = tf.compat.v1.distributions.Categorical(probs=probs, dtype=tf.float32)

        critic_value = critic(state)
        critic_value = tf.squeeze(critic_value)

        new_probs = dist.log_prob(action)
        prob_ratio = tf.math.exp(new_probs - old_probs)
        weighted_probs = advantage * prob_ratio
        clipped_probs = tf.clip_by_value(prob_ratio, 1-self.clip, 1+self.clip)
        weighted_clipped_probs = clipped_probs * advantage[batch]

        actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
        actor_loss = tf.math.reduce_mean(actor_loss)

        critic_loss = tf.keras.losses.mean_squared_error(critic_value, ret)

        return actor_loss, critic_loss

    def train_origin(self, batch, batches, origin_adv, origin_returns):
        with tf.GradientTape() as origin_actor_tape, tf.GradientTape() as origin_critic_tape:
            origin_actor_loss, origin_critic_loss = self.get_loss(self.origin_actor, self.origin_critic, batches, batch, origin_adv, origin_returns[batch], 0)
            
        origin_actor_params = self.origin_actor.trainable_variables
        origin_critic_params = self.origin_critic.trainable_variables
        origin_actor_grads = origin_actor_tape.gradient(origin_actor_loss, origin_actor_params)
        origin_critic_grads = origin_critic_tape.gradient(origin_critic_loss, origin_critic_params)
        self.origin_actor.optimizer.apply_gradients(zip(origin_actor_grads, origin_actor_params))
        self.origin_critic.optimizer.apply_gradients(zip(origin_critic_grads, origin_critic_params))

    def train_shell(self, batch, batches, shell_adv, shell_returns):
        with tf.GradientTape() as shell_actor_tape, tf.GradientTape() as shell_critic_tape:
            shell_actor_loss, shell_critic_loss = self.get_loss(self.shell_actor, self.shell_critic, batches, batch, shell_adv, shell_returns[batch], 1, shell=True)
        
        shell_actor_params = self.shell_actor.trainable_variables
        shell_critic_params = self.shell_critic.trainable_variables
        shell_actor_grads = shell_actor_tape.gradient(shell_actor_loss, shell_actor_params)
        shell_critic_grads = shell_critic_tape.gradient(shell_critic_loss, shell_critic_params)
        self.shell_actor.optimizer.apply_gradients(zip(shell_actor_grads, shell_actor_params))
        self.shell_critic.optimizer.apply_gradients(zip(shell_critic_grads, shell_critic_params))
    
    def train_card(self, batch, batches, card_adv, card_returns):
        with tf.GradientTape() as card_actor_tape, tf.GradientTape() as card_critic_tape:
            card_actor_loss, card_critic_loss = self.get_loss(self.card_actor, self.card_critic, batches, batch, card_adv, card_returns[batch], 2)
        
        card_actor_params = self.card_actor.trainable_variables
        card_critic_params = self.card_critic.trainable_variables
        card_actor_grads = card_actor_tape.gradient(card_actor_loss, card_actor_params)
        card_critic_grads = card_critic_tape.gradient(card_critic_loss, card_critic_params)
        self.card_actor.optimizer.apply_gradients(zip(card_actor_grads, card_actor_params))
        self.card_critic.optimizer.apply_gradients(zip(card_critic_grads, card_critic_params))

    def learn(self):
        batches = list(self.mem.generate_batches())

        origin_vals, shell_vals, card_vals, rewards, dones= self.get_adv_vals(batches)
        origin_adv, origin_returns = self.get_adv(origin_vals, rewards, dones)
        shell_adv, shell_returns= self.get_adv(shell_vals, rewards, dones)
        card_adv, card_returns = self.get_adv(card_vals, rewards, dones)

        for batch in range(len(batches)):
            self.train_origin(batch, batches, origin_adv, origin_returns)
            self.train_shell(batch, batches, shell_adv, shell_returns)  
            self.train_card(batch, batches, card_adv, card_returns)
        
        self.mem.clear()
    
    def train(self):
        for _ in range(self.epohs):
            self.learn()




if __name__ == '__main__':
    agent = Agent()
    env = Handler()
    state = env.get_state()
    agent.act(env, state)
    print("done")
        