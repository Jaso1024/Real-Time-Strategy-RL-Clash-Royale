from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import cv2
from keras.layers.activation import LeakyReLU
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class ElixirModel():
    def __init__(self):
        pass

    def create_model(self):
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

        data = pd.read_pickle("TrainingData/ElixirData.pkl")
        x_train, y_train = data["image"], data["Elixir"]

        x_train = np.array(x_train)
        y_train = np.array(y_train)


        normalized_x = []
        for image in x_train:
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            normalized_x.append(np.asarray(image))


        self.model.fit(np.array([*normalized_x]), np.array([*y_train]),
                       steps_per_epoch=len(normalized_x)/10,
                       epochs=1000,
                       verbose=1,
                       batch_size=10)


        self.save_model()






    def save_model(self):
        self.model.save("Resources/Models/Saved/ElixirModel")

    def load_model(self):
        self.model = keras.models.load_model("Resources/Models/Saved/ElixirModel")

    def predict(self, image):
        image = np.array(cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        prediction = self.model.predict(np.array([image]), verbose=0)
        return prediction



if __name__ == "__main__":
    model = ElixirModel()
    model.create_model()
    model.train()
    model.save_model()
    model.load_model()


