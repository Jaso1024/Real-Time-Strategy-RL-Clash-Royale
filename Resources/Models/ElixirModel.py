from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Input, BatchNormalization, Conv1D
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ElixirModel:
    """A model which predicts elixir count."""
    def __init__(self):
        self.model = None

    def create_model(self):
        """
        Builds a new model.

        :return: None
        """
        self.model = Sequential()
        self.model.add(Input(shape=(1, 167, 3)))
        self.model.add(Conv1D(16, kernel_size=5, strides=4, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(1, activation="relu"))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.000001),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=["mae"]
                           )

        self.model.summary()

    def train(self):
        """
        Trains the Sequential model.

        :return: None
        """

        data = pd.read_pickle("TrainingData/ElixirData.pkl")
        x_train, y_train = data["image"], data["Elixir"]

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        normalized_x = []
        for image in x_train:
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            normalized_x.append(np.asarray(image))

        self.model.fit(np.array([*normalized_x]), np.array([*y_train]),
                       steps_per_epoch=len(normalized_x) / 10,
                       epochs=1000,
                       verbose=1,
                       batch_size=10)

        self.save_model()

    def save_model(self):
        """
        Saves the current model in the "Resources/Models/Saved" folder.

        :return: None
        """
        self.model.save("Resources/Models/Saved/ElixirModel")

    def load_model(self):
        """
        Loads the model from the "Resources/Models/Saved" folder.

        :return: None
        """
        self.model = load_model("Resources/Models/Saved/ElixirModel")

    def predict(self, image):
        """
        Predicts the amount of elixir available given an image.

        :param image: A 2D iterable representing an image of the elixir bar
        :return: prediction (float): A float representing the amount of elixir available
        """
        image = np.array(cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        prediction = self.model.predict(np.array([image]), verbose=0)
        return prediction


if __name__ == "__main__":
    model = ElixirModel()
    model.create_model()
    model.train()
    model.save_model()
    model.load_model()
