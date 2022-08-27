from argparse import Action
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, concatenate, Flatten, LeakyReLU, BatchNormalization
from keras.activations import *
from keras import backend as K
from ClashRoyaleHandler import ClashRoyaleHandler


class BattleModel(Model):
    initializer = tf.keras.initializers.HeNormal()
    fully_connected_layers = [
        [
            Flatten(name="Elixir_layer_1-Flatten"),
            Dense(64, activation="relu", name="Elixir_layer_2-Dense")
        ],
        [
            Dense(9, activation="relu", name="Card_layer_1-Dense"),
            Flatten(name="Card_layer_2-Flatten"),
            Dense(128, activation='relu', name="Card_layer_3-Dense")
        ],
        [
            Conv2D(256, 5, padding="same", activation=None, name="Field_player_layer_1-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(64, 3, strides=2, padding="same", activation=None, name="Field_player_layer_2-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, 3, strides=2, padding="same", activation=None, name="Field_player_layer_3-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, 3, strides=2, padding="same", activation=None, name="Field_player_layer_3-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Flatten(name="Field_player_layer_4-Flatten")
        ],
        [
            Conv2D(256, 5, padding="same", activation=None, name="Field_player_layer_1-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(64, 3, strides=2, padding="same", activation=None, name="Field_enemy_layer_2-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, 3, strides=2, padding="same", activation=None, name="Field_enemy_layer_3-Conv2D"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, 3, strides=2, padding="same", activation=None, name="Field_enemy_layer_3-Conv2D"),
             BatchNormalization(),
            LeakyReLU(0.2),
            Flatten(name="Field_enemy_layer_4-Flatten")
        ],
        [
            Flatten(name="Field_left_layer_5-Flatten"),
            Dense(32, activation="relu", name="Field_left_layer_6-Dense")
        ],
        [
            Flatten(name="Field_right_layer_5-Flatten"),
            Dense(32, activation="relu", name="Field_right_layer_6-Dense")
        ],
    ]
    combined_layers = [
        Dense(2048, activation=None, name="Combined_factors_layer_1-Dense"),
        LeakyReLU(0.2)
    ]

    def call_fc_layers(cls, inputs):
        split_outs = []
        for inp, layers in zip(inputs, BattleModel.fully_connected_layers):
            x = layers[0](inp)
            for layer in layers[1:]:
                x = layer(x)
            split_outs.append(x)

        x = concatenate(split_outs)
        for layer in BattleModel.combined_layers:
            x = layer(x)
        return x
    
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


class OriginModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_squares_1 = Dense(49, activation=None)
        self.origin_squares_2 = LeakyReLU(0.2)
    
    def call(self, inp):
        x = self.origin_squares_1(inp)
        x = self.origin_squares_2(x)
        return x

