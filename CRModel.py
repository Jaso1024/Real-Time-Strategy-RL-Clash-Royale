from sqlite3 import adapt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, concatenate, Flatten, LeakyReLU, BatchNormalization, Reshape
from keras.activations import *
from keras.optimizers import *

from CRHandler import Handler
import time


class StateAutoEncoder(Model):
    """A model for Clash Royale battles"""

    def __init__(self):
        super(StateAutoEncoder, self).__init__()
        self.build_layers()

    def build_layers(self):
        initializer = tf.keras.initializers.HeNormal()

        # Encoder
        # Elixir
        self.elixir_1 = Flatten(name="Elixir_layer_1-Flatten")
        self.elixir_2 = Dense(64, activation="relu", name="Elixir_layer_2-Dense", kernel_initializer=initializer)

        # Cards
        self.card_1 = Dense(9, activation="relu", name="Card_layer_1-Dense", kernel_initializer=initializer)
        self.card_2 = Flatten(name="Card_layer_2-Flatten")
        self.card_3 = Dense(128, activation='relu', name="Card_layer_3-Dense", kernel_initializer=initializer)

        # Field
        self.field_player_1 = Conv2D(256, 5, padding="same", activation="relu", name="Field_player_layer_1-Conv2D", kernel_initializer=initializer)
        self.field_player_2 = BatchNormalization()
        self.field_player_3 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="Field_player_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_player_4 = BatchNormalization()
        self.field_player_5 = Conv2D(128, 3, strides=2, padding="same", activation="relu", name="Field_player_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_player_6 = BatchNormalization()
        self.field_player_7 = Conv2D(128, 3, strides=2, padding="same", activation="relu", name="Field_player_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_player_8 = BatchNormalization()
        self.field_player_9 = Flatten(name="Field_player_layer_4-Flatten")

        self.field_enemy_1 = Conv2D(256, 5, padding="same", activation="relu", name="Field_player_layer_1-Conv2D", kernel_initializer=initializer)
        self.field_enemy_2 = BatchNormalization()
        self.field_enemy_3 = Conv2D(64, 3, strides=2, padding="same", activation="relu", name="Field_enemy_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_enemy_4 = BatchNormalization()
        self.field_enemy_5 = Conv2D(128, 3, strides=2, padding="same", activation="relu", name="Field_enemy_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_enemy_6 = BatchNormalization()
        self.field_enemy_7 = Conv2D(128, 3, strides=2, padding="same", activation="relu", name="Field_enemy_layer_3-Conv2D", kernel_initializer=initializer)
        self.field_enemy_8 = BatchNormalization()
        self.field_enemy_9 = Flatten(name="Field_enemy_layer_4-Flatten")

        self.bridge_1 = Flatten(name="Field_layer_5-Flatten")
        self.bridge_2 = Dense(32, activation="relu", name="Field_layer_6-Dense", kernel_initializer=initializer)

        # combined
        self.combined_1 = Dense(4096, activation="relu", name="Combined_factors_layer_1-Dense", kernel_initializer=initializer)

        # decoder
        self.elixir_1_d = Dense(1, activation="relu")
        self.card_1_d = Dense(9, activation="sigmoid")
        self.field_player_1_d = Dense(1683, activation="relu")
        self.field_player_2_d = Reshape((33, 51))
        self.field_enemy_1_d = Dense(1632, activation="relu")
        self.field_enemy_2_d = Reshape((32, 51))
        self.bridge_l_1_d = Dense(24, activation="relu")
        self.bridge_r_1_d = Dense(24, activation="relu")
        self.bridge_2_d = Reshape((6,4))



    def encode(self, inputs):
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
        inputs = self.format_data(inputs)
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = inputs
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
        field_p_x = self.field_player_6(field_p_x)
        field_p_x = self.field_player_7(field_p_x)
        field_p_x = self.field_player_8(field_p_x)
        field_p_x = self.field_player_9(field_p_x)

        field_e_x = self.field_enemy_1(field_e_in)
        field_e_x = self.field_enemy_2(field_e_x)
        field_e_x = self.field_enemy_3(field_e_x)
        field_e_x = self.field_enemy_4(field_e_x)
        field_e_x = self.field_enemy_5(field_e_x)
        field_e_x = self.field_enemy_6(field_e_x)
        field_e_x = self.field_enemy_7(field_e_x)
        field_e_x = self.field_enemy_8(field_e_x)
        field_e_x = self.field_enemy_9(field_e_x)

        field_l_x = self.bridge_1(field_l_in)
        field_l_x = self.bridge_2(field_l_x)

        field_r_x = self.bridge_1(field_r_in)
        field_r_x = self.bridge_2(field_r_x)

        x = concatenate([elixir_x, card_1x, card_2x, card_3x, card_4x, field_p_x, field_e_x, field_l_x, field_r_x])
        x = self.combined_1(x)
        return x
    
    def call(self, inputs):
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = inputs
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
        field_p_x = self.field_player_6(field_p_x)
        field_p_x = self.field_player_7(field_p_x)
        field_p_x = self.field_player_8(field_p_x)
        field_p_x = self.field_player_9(field_p_x)

        field_e_x = self.field_enemy_1(field_e_in)
        field_e_x = self.field_enemy_2(field_e_x)
        field_e_x = self.field_enemy_3(field_e_x)
        field_e_x = self.field_enemy_4(field_e_x)
        field_e_x = self.field_enemy_5(field_e_x)
        field_e_x = self.field_enemy_6(field_e_x)
        field_e_x = self.field_enemy_7(field_e_x)
        field_e_x = self.field_enemy_8(field_e_x)
        field_e_x = self.field_enemy_9(field_e_x)

        field_l_x = self.bridge_1(field_l_in)
        field_l_x = self.bridge_2(field_l_x)

        field_r_x = self.bridge_1(field_r_in)
        field_r_x = self.bridge_2(field_r_x)

        x = concatenate([elixir_x, card_1x, card_2x, card_3x, card_4x, field_p_x, field_e_x, field_l_x, field_r_x])
        x = self.combined_1(x)
        e  = self.elixir_1_d(x)

        c1 = self.card_1_d(x)
        c2 = self.card_1_d(x)
        c3 = self.card_1_d(x)
        c4 = self.card_1_d(x)


        fp = self.field_player_1_d(x)
        fp = self.field_player_2_d(fp)

        fe = self.field_enemy_1_d(x)
        fe = self.field_enemy_2_d(fe)

        bleft = self.bridge_l_1_d(x)
        bleft = self.bridge_2_d(bleft)

        bright = self.bridge_r_1_d(x)
        bright = self.bridge_2_d(bright)

        return e, [[c1, c2, c3, c4]], fp, fe, bleft, bright

    def predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = self.format_data(x)
        return super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = self.format_data(x)
        return super().fit(x, x, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    
    def format_data(self, state_data):
        """Formats the inputs"""
        def reshape_choice_data(state_data, data):
            card_data = []
            for card in state_data["card_data"]:
                card_data.append(np.expand_dims(card[1], axis=0))
            data.append([card_data])
            return data

        def reshape_field_data(state_data, data):
            player_side = np.array(state_data["field_data"]["player_side_dimensions"], dtype=np.float32).reshape((1, 33, 51, 1))
            player_side = self.normalize_img(player_side)
            enemy_side = np.array(state_data["field_data"]["enemy_side_dimensions"], dtype=np.float32).reshape((1, 32, 51, 1))
            enemy_side = self.normalize_img(enemy_side)
            left_side = np.array(state_data["field_data"]["left_bridge_dimensions"], dtype=np.float32).reshape((1, 6, 4, 1))
            left_side = self.normalize_img(left_side)
            right_side = np.array(state_data["field_data"]["right_bridge_dimensions"], dtype=np.float32).reshape((1, 6, 4, 1))
            right_side = self.normalize_img(right_side)
            data.extend([player_side, enemy_side, left_side, right_side])
            return data

        data = []
        data.append(np.array(float(state_data["elixir_data"])).reshape((1)))
        data = reshape_choice_data(state_data, data)
        data = reshape_field_data(state_data, data) 
        return data
    
    def normalize_img(self, img):
        """Normalize the given images values between 0 and 1"""
        return img/255

class Critic(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c1 = Dense(1, activation="relu")
        self.c2 = LeakyReLU(0.2)
    
    def call(self, inp):
        v = self.c1(inp)
        v = self.c2(v)
        return v

    def predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = StateAutoEncoder.format_data(x)
        return super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        x = StateAutoEncoder.format_data(x)
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    
class OriginActor(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_squares_1 = Dense(49, activation='softmax')
    
    def call(self, inp):
        x = self.origin_squares_1(inp)
        return x

class TileActor(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tile_2 = Dense(9, activation="softmax")
    
    def call(self, inp):
        shell_tile = self.tile_2(inp)
        return shell_tile

class CardActor(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.card_1 = Dense(4, activation="softmax")
    
    def call(self, inp):
        card = self.card_1(inp)
        return card

if __name__ == '__main__':
    ae = StateAutoEncoder()
    env = Handler()
    state = env.get_state()
    opt = Adam(learning_rate=1e-5)
    ae.compile(optimizer=opt, loss="mse")
    start = time.time()
    print(ae.fit(state, state, epochs=5))
    print(time.time()-start)
