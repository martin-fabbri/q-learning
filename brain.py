# imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


class Brain(object):
    def __init__(self, learning_rate=0.001, num_actions=5):
        self.learning_rate = learning_rate
        states = Input(shape=(3,))
        x = Dense(units=64, activation='sigmoid')(states)
        y = Dense(units=32, activation='sigmoid')(x)
        q_values = Dense(units=num_actions, activation='softmax')(y)
        self.model = Model(inputs=states, outputs=q_values)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
