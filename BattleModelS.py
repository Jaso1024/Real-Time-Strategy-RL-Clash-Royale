
from turtle import left, right
from cv2 import CV_32F
from ClashRoyaleHandler import ClashRoyaleHandler
from ReplayBuffer import ReplayBuffer

import cv2
import numpy as np
import math
from deepdiff import diff

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.callbacks import LearningRateScheduler
from keras.models import load_model, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers import Conv2D, Dense, concatenate, Input, MaxPooling2D, Flatten
from keras import backend as K

import os
import absl.logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)


class BattleModel(Model):
    def __init__(self):
        super(BattleModel, self).__init__()

        initializer = tf.keras.initializers.HeNormal()
        
        self.elixir_1 = Flatten(name="Elixir_layer_2-Flatten")

        # Cards
        self.card_1 = Conv2D(32, 3, padding="same", activation="relu", name="Cards_layer_2-Conv2D", kernel_initializer=initializer)
        self.card_2 = Flatten(name="Cards_layer_5-Flatten")
        self.card_3 = Dense(256, activation="relu", name="Cards_layer_6-Dense", kernel_initializer=initializer)

        # field
        self.field_player_1 = Conv2D(16, 8, strides=4, padding="same", activation="relu",name="Field_player_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_player_2 = Conv2D(32, 4, strides=2, padding="same", activation="relu",name="Field_player_layer_2.5-Conv2", kernel_initializer=initializer)
        self.field_player_3 = Conv2D(64, 3, strides=1, padding="same", activation="relu",name="Field_player_layer_5-Conv2", kernel_initializer=initializer)
        self.field_player_4 = Flatten(name="Field_player_layer_6-Flatten")
        self.field_player_5 = Dense(512, activation="relu", name="Field_player_layer_8-Dense", kernel_initializer=initializer)

        self.field_enemy_1 = Conv2D(16, 8, strides=4, padding="same", activation="relu",name="Field_enemy_layer_2-Conv2D", kernel_initializer=initializer)
        self.field_enemy_2 = Conv2D(32, 4, strides=2, padding="same", activation="relu",name="Field_enemy_layer_4-Conv2D", kernel_initializer=initializer)
        self.field_enemy_3 = Conv2D(64, 3, strides=1, padding="same", activation="relu", name="Field_enemy_layer_5-Conv2D", kernel_initializer=initializer)
        self.field_enemy_4 = Flatten(name="Field_enemy_layer_6-Flatten")
        self.field_enemy_5 = Dense(512, activation="relu", name="Field_enemy_layer_7-Dense", kernel_initializer=initializer)

        self.field_left_1 = Conv2D(32, 3, strides=1, padding="same", activation="relu", kernel_initializer=initializer)
        self.field_left_2 = Flatten(name="Field_left_layer_5-Flatten")
        self.field_left_3 = Dense(256, activation="relu", name="Field_left_layer_6-Dense", kernel_initializer=initializer)

        self.field_right_1 = Conv2D(32, 3, strides=1, padding="same", activation="relu", kernel_initializer=initializer)
        self.field_right_2 = Flatten(name="Field_right_layer_5-Flatten")
        self.field_right_3 = Dense(256, activation="relu", name="Field_right_layer_6-Dense", kernel_initializer=initializer)

        # combined
        self.combined_1 = Dense(512, activation="relu", name="Combined_factors_layer_1-Dense", kernel_initializer=initializer)
        self.combined_2 = Dense(512, activation="relu", name="Combined_factors_layer_2-Dense", kernel_initializer=initializer)
        
        # State value
        self.state_val = Dense(1, activation='linear', kernel_initializer=initializer)
        
        # Advantage value
        self.advantage_val = Dense(2017, activation="relu", name="Output_layer", kernel_initializer=initializer)

    def predict_on_batch(self, x):
        state = self.format_data(x)
        return super().predict_on_batch(state)
    
    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True, return_dict=False):
        state = self.format_data(x)
        target = self.normalize_target(y)
        return super().train_on_batch(state, target, sample_weight, class_weight, reset_metrics, return_dict)

    def predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        state = self.format_data(x)
        return super().predict(state, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        state = self.format_data(x)
        target = self.normalize_target(y)
        return super().fit(state, target, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    
    def call_combined(self, elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in):

        elixir_x = self.elixir_1(elixir_in)

        card_x = self.card_1(card_in)
        card_x = self.card_2(card_x)
        card_x = self.card_3(card_x)

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

        x = concatenate([elixir_x, card_x, field_p_x, field_e_x, field_l_x, field_r_x])
        x = self.combined_1(x)
        x = self.combined_2(x)

        return x

    def call(self, state):
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = state
        x = self.call_combined(elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in)
        s = self.state_val(x)
        a = self.advantage_val(x)

        Q = s + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in = self.format_data(state)
        x = self.call_combined(elixir_in, card_in, field_p_in, field_e_in, field_l_in, field_r_in)
        a = self.advantage_val(x)

        checker = np.array(a).flatten()
        return a
    
    def normalize_img(self, img):
        return img/255
    
    def normalize_target(self, target):
        max = 116.0
        min = -115.0
        target += 115.0
        target = (target/231.0) * 2
        target = target - 1.0
        return target

    def format_data(self, state_data):
        def reshape_choice_data(state_data, data):
            card_images = []
            same_shape_card_images = []
            for item in state_data["choice_data"]:
                if item is None:
                    continue
                elif item["card_image"].shape[-1] == 3:
                    continue
                elif np.array([np.equal(item["card_image"], card).all() for card in same_shape_card_images]).any():
                    continue
                else:
                    card = np.array(item["card_image"]).reshape(43, 35, 1)
                    card = self.normalize_img(card)
                    card_images.append(card)
                    same_shape_card_images.append(item["card_image"])

            data.append(np.array(card_images, dtype=np.float32).reshape((1,4,43,35,1)))
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
    env = ClashRoyaleHandler()
    state = env.get_state()
    model = BattleModel()
    print(model.predict(state))

