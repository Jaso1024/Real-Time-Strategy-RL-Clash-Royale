import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, concatenate, Flatten
from ClashRoyaleHandler import ClashRoyaleHandler


class BattleModel(Model):
    """A model for Clash Royale battles"""
    def __init__(self):
        super(BattleModel, self).__init__()
        self.build_layers()

    def build_layers(self):
        initializer = tf.keras.initializers.HeNormal()

        self.elixir_1 = Flatten(name="Elixir_layer_1-Flatten")
        self.elixir_2 = Dense(64, activation="relu", name="Elixir_layer_2-Dense", kernel_initializer=initializer)

        # Cards
        self.card_1 = Dense(9, activation="relu", name="Card_layer_1-Dense", kernel_initializer=initializer)
        self.card_2 = Flatten(name="Card_layer_2-Flatten")
        self.card_3 = Dense(128, activation='relu', name="Card_layer_3-Dense", kernel_initializer=initializer)

        # field
        self.field_player_1 = Conv2D(32, 5, strides=4, padding="same", activation="relu", name="Field_player_layer_1-Conv2D", kernel_initializer=initializer)
        self.field_player_2 = Conv2D(32, 3, strides=2, padding="same", activation="relu", name="Field_player_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_player_3 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="Field_player_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_player_4 = Flatten(name="Field_player_layer_4-Flatten")
        self.field_player_5 = Dense(128, activation="relu", name="Field_player_layer_5-Dense", kernel_initializer=initializer)

        self.field_enemy_1 = Conv2D(32, 5, strides=4, padding="same", activation="relu", name="Field_enemy_layer_1-Conv2D", kernel_initializer=initializer)
        self.field_enemy_2 = Conv2D(32, 3, strides=2, padding="same", activation="relu", name="Field_enemy_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_enemy_3 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="Field_enemy_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_enemy_4 = Flatten(name="Field_enemy_layer_4-Flatten")
        self.field_enemy_5 = Dense(128, activation="relu", name="Field_enemy_layer_5-Dense", kernel_initializer=initializer)

        self.field_left_1 = Conv2D(16, 5, strides=2, padding="same", activation="relu", kernel_initializer=initializer)
        self.field_left_2 = Flatten(name="Field_left_layer_5-Flatten")
        self.field_left_3 = Dense(32, activation="relu", name="Field_left_layer_6-Dense", kernel_initializer=initializer)

        self.field_right_1 = Conv2D(16, 5, strides=2, padding="same", activation="relu", kernel_initializer=initializer)
        self.field_right_2 = Flatten(name="Field_right_layer_5-Flatten")
        self.field_right_3 = Dense(32, activation="relu", name="Field_right_layer_6-Dense", kernel_initializer=initializer)

        # combined
        self.combined_1 = Dense(1024, activation="relu", name="Combined_factors_layer_1-Dense", kernel_initializer=initializer)
        
        # State value
        self.state_val = Dense(1, activation="linear", kernel_initializer=initializer)
        
        # Advantage value
        self.advantage_val = Dense(48, activation="linear", name="Output_layer", kernel_initializer=initializer)
    
    def predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = self.format_data(x)
        return super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = self.format_data(x)
        y = self.normalize_target(y)
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    
    def call_combined(self, elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in):
        """
        Calls the layers that are shared between the state and advantage values.

        :param elixir_in: An integer representing the amount of elixir available
        :param card_in: A list of numpy arrays representing the cards available
        :param field_p_in: A 2D tensor representing an image of the player's side of the field
        :param field_e_in: A 2D tensor representing an image of the enemy's side of the field
        :param field_l_in: A 2D tensor representing an image of the left bridge
        :param field_r_in: A 2D tensor representing an image of the right bridge
        :return: x (tf.tensor): A tensor containing the output values (1024) of the last combined layer
        """
        elixir_x = self.elixir_1(elixir_in)
        elixir_x = self.elixir_2(elixir_x)

        card_1x = self.card_1(card_in[0][0])
        card_1x = self.card_2(card_1x)
        card_1x = self.card_3(card_1x)

        card_2x = self.card_1(card_in[0][1])
        card_2x = self.card_2(card_2x)
        card_2x = self.card_3(card_2x)

        card_3x = self.card_1(card_in[0][2])
        card_3x = self.card_2(card_3x)
        card_3x = self.card_3(card_3x)

        card_4x = self.card_1(card_in[0][3])
        card_4x = self.card_2(card_4x)
        card_4x = self.card_3(card_4x)

        field_p_x = self.field_player_1(field_p_in)
        field_p_x = self.field_player_2(field_p_x)
        field_p_x = self.field_player_3(field_p_x)
        field_p_x = self.field_player_4(field_p_x)
        field_p_x = self.field_player_5(field_p_x)

        field_e_x = self.field_enemy_1(field_e_in)
        field_e_x = self.field_enemy_2(field_e_x)
        field_e_x = self.field_enemy_3(field_e_x)
        field_e_x = self.field_enemy_4(field_e_x)
        field_e_x = self.field_enemy_5(field_e_x)

        field_l_x = self.field_left_1(field_l_in)
        field_l_x = self.field_left_2(field_l_x)
        field_l_x = self.field_left_3(field_l_x)

        field_r_x = self.field_left_1(field_r_in)
        field_r_x = self.field_left_2(field_r_x)
        field_r_x = self.field_left_3(field_r_x)

        x = concatenate([elixir_x, card_1x, card_2x, card_3x, card_4x, field_p_x, field_e_x, field_l_x, field_r_x])
        x = self.combined_1(x)

        return x

    def call(self, inputs):
        """
        Returns the value of being in the current state (the input to the model).

        :param inputs: A Tensor containing data about the elixir, cards, and field
        :return: q (float): The value of being in the current state
        """
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = inputs
        x = self.call_combined(elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in)
        s = self.state_val(x)
        a = self.advantage_val(x)
        q = s + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return q

    def advantage(self, inputs):
        """
        Returns the value of taking a certain action from the current state.

        :param inputs: A Tensor containing data about the elixir, cards, and field
        :return: a (tf.tensor): A tensor containing the values of each action
        """
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = self.format_data(inputs)
        x = self.call_combined(elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in)
        a = self.advantage_val(x)
        return a
    
    def normalize_img(self, img):
        """Normalize the given images values between 0 and 1"""
        return img/255
    
    def normalize_target(self, target):
        """normalizes the target between 0 and 1"""
        return target

    def format_data(self, state_data):
        """Formats the inputs"""
        def reshape_choice_data(state_data, data):
            card_data = []
            for card in state_data["card_data"]:
                card_data.append(np.expand_dims(card[1], axis=0))
            data.append([card_data])
            return data

        def reshape_field_data(state_data, data):
            player_side = np.array(state_data["field_data"]["player_side_dimensions"], dtype=np.float32).reshape((1, 132, 206, 1))
            player_side = self.normalize_img(player_side)
            enemy_side = np.array(state_data["field_data"]["enemy_side_dimensions"], dtype=np.float32).reshape((1, 128, 206, 1))
            enemy_side = self.normalize_img(enemy_side)
            left_side = np.array(state_data["field_data"]["left_bridge_dimensions"], dtype=np.float32).reshape((1, 24, 18, 1))
            left_side = self.normalize_img(left_side)
            right_side = np.array(state_data["field_data"]["right_bridge_dimensions"], dtype=np.float32).reshape((1, 24, 18, 1))
            right_side = self.normalize_img(right_side)
            data.extend([player_side, enemy_side, left_side, right_side])
            return data

        data = []
        data.append(np.array(float(state_data["elixir_data"])).reshape((1)))
        data = reshape_choice_data(state_data, data)
        data = reshape_field_data(state_data, data) 
        return data


if __name__ == "__main__":
    model = BattleModel()
    env = ClashRoyaleHandler()
    state = env.get_state()
    model.predict(state)
    model.summary()