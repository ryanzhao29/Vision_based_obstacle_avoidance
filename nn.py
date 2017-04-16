"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

# Adding this per a suggestion by Tim Kelch.
# https://medium.com/@trkelch/this-post-is-great-possibly-the-best-tutorial-explanation-ive-found-thus-far-cf78886b5378#.w473ywtbw
import tensorflow as tf
# tf.python.control_flow_ops = tf


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def neural_net(num_sensors, params, load='', dropout = True):
    model = Sequential()

    # First layer.
    model.add(Dense(
        params[0], init='lecun_uniform', input_shape=(7,)
    ))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.3))

    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.3))

    # Output layer.
    model.add(Dense(5, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model


def lstm_net(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(
        output_dim=64, input_dim=num_sensors, return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=64, input_dim=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=5, input_dim=64))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model
